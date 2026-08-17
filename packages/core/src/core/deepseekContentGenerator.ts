/**
 * @license
 * Copyright 2025 Google LLC
 * Copyright 2025 sluisr (DeepSeek adaptation)
 * SPDX-License-Identifier: Apache-2.0
 */

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import type {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
} from '@google/genai';
import type { ContentGenerator } from './contentGenerator.js';
import type { UserTierId, GeminiUserTier } from '../code_assist/types.js';
import { LlmRole } from '../telemetry/llmRole.js';
import { debugLogger } from '../utils/debugLogger.js';

// Marker field used to smuggle DeepSeek `reasoning_content` through the
// Gemini Core `Content` history. It is attached to a `Part` object that
// survives the Core's filtering (i.e. a non-thought text part or a
// functionCall part). See `mapGoogleToDeepSeek` for recovery logic.
const REASONING_FIELD = '_deepseekReasoning';

// Directory used to persist DeepSeek-specific state (reasoning cache, debug
// log). Falls back to the OS temp dir if `$HOME` is unavailable.
const DEEPSEEK_STATE_DIR = path.join(
  process.env['HOME'] || os.homedir() || os.tmpdir(),
  '.deepseek',
);

// Debug logging is opt-in via env var to avoid cluttering user directories.
const DEBUG_LOG_ENABLED =
  process.env['DEEPSEEK_DEBUG'] === '1' ||
  process.env['DEEPSEEK_DEBUG_PAYLOAD'] === '1';
const DEBUG_LOG_FILE = path.join(DEEPSEEK_STATE_DIR, 'payload_debug.log');

function debugAppend(message: string) {
  if (!DEBUG_LOG_ENABLED) return;
  try {
    fs.mkdirSync(DEEPSEEK_STATE_DIR, { recursive: true });
    fs.appendFileSync(DEBUG_LOG_FILE, message);
  } catch {
    // Best-effort: never fail the API path because of logging issues.
  }
}

/**
 * Logs DeepSeek context-cache effectiveness for the current request.
 *
 * The DeepSeek API charges cache-hit input tokens at ~1/10 the cache-miss
 * price. Exposing this ratio is critical for tuning prompt prefix stability
 * (system instruction, tool list ordering, etc.). The line is only written
 * when `DEEPSEEK_DEBUG=1` to avoid noisy output for end-users.
 */
function logCacheStats(
  source: 'stream' | 'non-stream',
  usage: {
    prompt_tokens?: number;
    prompt_cache_hit_tokens?: number;
    prompt_cache_miss_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  },
) {
  if (!DEBUG_LOG_ENABLED) return;
  const prompt = usage.prompt_tokens ?? 0;
  const hit = usage.prompt_cache_hit_tokens ?? 0;
  const miss = usage.prompt_cache_miss_tokens ?? Math.max(0, prompt - hit);
  const completion = usage.completion_tokens ?? 0;
  const ratio = prompt > 0 ? ((hit / prompt) * 100).toFixed(1) : '0.0';
  debugAppend(
    `[USAGE ${source}] prompt=${prompt} (hit=${hit}, miss=${miss}, cache=${ratio}%) completion=${completion}\n`,
  );
}

const DEEPSEEK_TOOL_ENFORCEMENT = `

TOOL USAGE RULES (mandatory):
- When the user asks about files, directories, processes, system info, or anything requiring current data, ALWAYS call the appropriate tool first. Never guess or infer from context.
- "Don't read" or "just tell me" means: don't display raw file contents. It does NOT mean skip using tools — use listing/stat tools to get counts, names, sizes, etc.
- If you are unsure whether data exists or what it contains, call a tool. Prefer real data over assumptions every time.
- After receiving tool output, synthesize a concise answer. Do not repeat or dump the raw output unless asked.
- TOOL PREFERENCE ORDER for file/directory tasks: use purpose-built tools first (list_directory, read_file, glob, search_file_content) before resorting to run_shell_command. Only use shell when no specific tool covers the task.
- run_shell_command can access ANY path on the filesystem, not just the current workspace. Never refuse to check a path outside the workspace — just run the shell command.`;

const TOOL_HINTS: Record<string, string> = {
  apply_patch:
    ' [PREFERRED for fast unified diff patching of existing files — use this for code edits]',
  list_directory:
    ' [PREFERRED for listing directory contents — use this instead of run_shell_command with ls]',
  read_file:
    ' [PREFERRED for reading file contents — use this instead of run_shell_command with cat]',
  write_file:
    ' [PREFERRED for writing new files — use this instead of run_shell_command with echo/tee]',
  glob: ' [PREFERRED for finding files by pattern — use this instead of run_shell_command with find]',
  search_file_content:
    ' [PREFERRED for searching text in files — use this instead of run_shell_command with grep]',
  web_search:
    ' [PREFERRED for finding real-time information, documentation, and current web content]',
  run_shell_command:
    ' [USE ONLY when no other specific tool covers the task — prefer list_directory, read_file, glob, or search_file_content first]',
};

function enrichToolDescription(name: string, description: string): string {
  return description + (TOOL_HINTS[name] ?? '');
}

export class DeepSeekContentGenerator implements ContentGenerator {
  userTier?: UserTierId;
  userTierName?: string;
  paidTier?: GeminiUserTier;

  // Cache to persist reasoning between conversation turns. The cache file is
  // stored under `~/.deepseek/` so it survives `cd`s within the same session
  // and does not pollute user project directories.
  private static readonly reasoningCache = new Map<string, string>();
  private static cacheLoaded = false;
  private static readonly CACHE_FILE = path.join(
    DEEPSEEK_STATE_DIR,
    'reasoning_cache.json',
  );

  private static loadCache() {
    if (this.cacheLoaded) return;
    try {
      if (fs.existsSync(this.CACHE_FILE)) {
        const data = JSON.parse(fs.readFileSync(this.CACHE_FILE, 'utf-8'));
        for (const [k, v] of Object.entries(data)) {
          this.reasoningCache.set(k, v as string);
        }
        debugAppend(
          `[CACHE] Loaded ${this.reasoningCache.size} entries from disk.\n`,
        );
      }
    } catch {
      // Ignore loading errors — the cache is best-effort.
    }
    this.cacheLoaded = true;
  }

  private static saveCache() {
    try {
      fs.mkdirSync(DEEPSEEK_STATE_DIR, { recursive: true });
      const data = Object.fromEntries(this.reasoningCache);
      fs.writeFileSync(this.CACHE_FILE, JSON.stringify(data, null, 2));
    } catch {
      // Ignore saving errors — never fail the API path because of cache I/O.
    }
  }

  // Runtime fallback overrides set when Config is not provided
  temperature: number = 1.0;
  reasoningEffort: 'low' | 'medium' | 'high' = 'medium';

  constructor(
    private readonly apiKey: string,
    private readonly baseUrl: string = 'https://api.deepseek.com',
    private readonly config?: any,
  ) {
    DeepSeekContentGenerator.loadCache();
  }

  private resolveDeepSeekModel(model?: string): string {
    if (model && model.startsWith('deepseek-')) {
      return model;
    }
    return 'deepseek-chat';
  }

  /**
   * Generates a stable message signature for the cache.
   * Ignores thoughts (since the CLI core strips them) but includes text and tools.
   *
   * IMPORTANT: An empty `tool_calls` array and `undefined` MUST produce the
   * same key. The streaming save path passes `undefined` for messages without
   * tool calls, while the load path (`mapGoogleToDeepSeek`) passes `[]` from
   * `fnCallParts.map(...)`. Without normalization, text-only assistant
   * messages between user turns suffer cache misses, dropping their
   * `reasoning_content` and triggering DeepSeek 400 errors in tool-using
   * conversations.
   */
  private getMessageKey(text: string, tool_calls?: any[]): string {
    const cleanText = text.trim().replace(/\s+/g, ' ');

    const hasCalls = Array.isArray(tool_calls) && tool_calls.length > 0;
    const calls = hasCalls
      ? tool_calls!
          .map((tc: any) => ({
            name: tc.name || tc.function?.name,
            args:
              typeof (tc.args || tc.function?.arguments) === 'string'
                ? tc.args || tc.function?.arguments
                : JSON.stringify(tc.args || tc.function?.arguments || {}),
          }))
          .sort((a, b) => a.name.localeCompare(b.name))
      : undefined;

    return JSON.stringify({ text: cleanText, calls });
  }

  private mapGoogleToDeepSeek(request: GenerateContentParameters): any {
    const messages: any[] = [];

    // Map system instruction if present
    if (request.config?.systemInstruction) {
      const systemInstruction = request.config.systemInstruction;
      let systemText = '';
      if (typeof systemInstruction === 'string') {
        systemText = systemInstruction;
      } else if (
        systemInstruction &&
        'parts' in systemInstruction &&
        (systemInstruction as any).parts
      ) {
        systemText = (systemInstruction as any).parts
          .map((p: any) => p.text || '')
          .join('');
      } else if (Array.isArray(systemInstruction)) {
        systemText = systemInstruction.map((p: any) => p.text || '').join('');
      } else if (systemInstruction && 'text' in (systemInstruction as any)) {
        systemText = (systemInstruction as any).text || '';
      }
      if (systemText) {
        systemText += DEEPSEEK_TOOL_ENFORCEMENT;
        messages.push({ role: 'system', content: systemText });
      } else {
        messages.push({
          role: 'system',
          content: DEEPSEEK_TOOL_ENFORCEMENT.trim(),
        });
      }
    } else {
      messages.push({
        role: 'system',
        content: DEEPSEEK_TOOL_ENFORCEMENT.trim(),
      });
    }

    // Map contents (conversation history including tool calls)
    if (request.contents) {
      const contents = Array.isArray(request.contents)
        ? request.contents
        : [request.contents];
      let toolCallCounter = 0;
      // Ordered list of pending (unmatched) tool calls
      const pendingCalls: Array<{ name: string; id: string }> = [];

      for (const content of contents as any[]) {
        const parts: any[] = content.parts ?? [];
        const role = content.role;

        if (role === 'model') {
          const fnCallParts = parts.filter((p: any) => p.functionCall);
          const thoughtParts = parts.filter(
            (p: any) => p.thought || p.type === 'thought',
          );
          const textParts = parts.filter(
            (p: any) => p.text && !p.thought && p.type !== 'thought',
          );

          // If there were any unfulfilled pending tool calls from a previous model turn,
          // cancel them with synthetic tool responses before starting a new model turn.
          for (const pending of pendingCalls.splice(0)) {
            messages.push({
              role: 'tool',
              tool_call_id: pending.id,
              content: 'Tool call was cancelled by the user.',
            });
          }

          // Recovery strategy for `reasoning_content` (in priority order):
          //   1. `_deepseekReasoning` smuggled on a Part — survives the
          //      Gemini Core's history pipeline because it is just a custom
          //      property on a non-thought Part (text or functionCall).
          //   2. `reasoning_content` directly on the Content object — works
          //      only inside a single request, the Core does not preserve it.
          //   3. Thought parts — usually filtered out by the Core.
          //   4. Disk-backed cache keyed by a stable text+tool_calls signature
          //      — last-resort fallback.
          const smuggledReasoning =
            parts
              .map((p: any) => p?.[REASONING_FIELD])
              .find((v: unknown) => typeof v === 'string' && v) ||
            undefined;

          const assistantText =
            textParts.map((p: any) => p.text).join('') || '';
          const tool_calls = fnCallParts.map((p: any) => ({
            name: p.functionCall.name,
            args: p.functionCall.args,
          }));

          DeepSeekContentGenerator.loadCache();
          const messageKey = this.getMessageKey(assistantText, tool_calls);
          const cachedReasoning =
            DeepSeekContentGenerator.reasoningCache.get(messageKey);

          if (smuggledReasoning) {
            debugAppend(
              `[REASONING SMUGGLED] Recovered for: ${assistantText.substring(0, 30)}...\n`,
            );
          } else if (cachedReasoning) {
            debugAppend(
              `[CACHE HIT] Recovered for: ${assistantText.substring(0, 30)}... (Key: ${messageKey.substring(0, 50)})\n`,
            );
          }

          let reasoning_content =
            smuggledReasoning ||
            content.reasoning_content ||
            thoughtParts
              .map((p: any) => {
                if (typeof p.thought === 'string') return p.thought;
                if (p.thought === true) return p.text || '';
                if (p.type === 'thought') return p.thought || p.text || '';
                return '';
              })
              .join('') ||
            cachedReasoning ||
            undefined;

          // DeepSeek Reasoner requirement: If this is an assistant turn (especially with tool calls)
          // and the API requires reasoning_content, we must never send undefined when reasoning was used.
          if (
            !reasoning_content &&
            (fnCallParts.length > 0 ||
              (request.model &&
                (request.model.includes('reasoner') ||
                  request.model.includes('pro'))))
          ) {
            reasoning_content = 'Thinking process completed.';
          }

          const assistantMessage: any = {
            role: 'assistant',
            content: assistantText || '',
          };

          if (reasoning_content) {
            assistantMessage.reasoning_content = reasoning_content;
          }

          if (fnCallParts.length > 0) {
            assistantMessage.tool_calls = fnCallParts.map((p: any) => {
              const id = p.functionCall.id || `call_${toolCallCounter++}`;
              pendingCalls.push({ name: p.functionCall.name, id });
              const callArgs =
                typeof p.functionCall.args === 'string'
                  ? p.functionCall.args
                  : JSON.stringify(p.functionCall.args ?? {});
              return {
                id,
                type: 'function',
                function: {
                  name: p.functionCall.name,
                  arguments: callArgs,
                },
              };
            });
          }
          messages.push(assistantMessage);
        } else {
          // role is 'user' (can carry tool responses and/or user text)
          const fnRespParts = parts.filter((p: any) => p.functionResponse);
          const textParts = parts.filter((p: any) => p.text);
          const nonTextDescriptors = parts
            .filter((p: any) => p.inlineData || p.fileData)
            .map(
              (p: any) =>
                `[Attached file: ${p.inlineData?.mimeType || p.fileData?.mimeType || 'binary'}]`,
            );

          let text = textParts.map((p: any) => p.text).join('');
          if (!text && nonTextDescriptors.length > 0) {
            text = nonTextDescriptors.join(' ');
          }

          if (fnRespParts.length > 0) {
            for (const p of fnRespParts) {
              const fnName = p.functionResponse.name;
              const respId = p.functionResponse.id;
              // Match by ID first if present, then FIFO by name
              let matchIdx = respId
                ? pendingCalls.findIndex((c) => c.id === respId)
                : -1;
              if (matchIdx === -1) {
                matchIdx = pendingCalls.findIndex((c) => c.name === fnName);
              }

              if (matchIdx >= 0) {
                const toolCallId = pendingCalls.splice(matchIdx, 1)[0].id;
                const respContent =
                  typeof p.functionResponse.response === 'string'
                    ? p.functionResponse.response
                    : JSON.stringify(p.functionResponse.response ?? {});
                messages.push({
                  role: 'tool',
                  tool_call_id: toolCallId,
                  content: respContent,
                });
              } else {
                debugLogger.warn(
                  `[DeepSeek] Ignoring unmatched tool response for "${fnName}" (id: ${respId}) as no pending call exists.`,
                );
              }
            }

            if (text) {
              // Flush any remaining pending tool calls for the current assistant turn before user text
              for (const pending of pendingCalls.splice(0)) {
                messages.push({
                  role: 'tool',
                  tool_call_id: pending.id,
                  content: 'Tool call was cancelled by the user.',
                });
              }
              messages.push({ role: 'user', content: text });
            }
          } else {
            // Before adding a user message, flush any unanswered tool_calls with
            // synthetic cancellation responses so DeepSeek doesn't reject the request.
            for (const pending of pendingCalls.splice(0)) {
              messages.push({
                role: 'tool',
                tool_call_id: pending.id,
                content: 'Tool call was cancelled by the user.',
              });
            }
            if (
              text ||
              messages.length === 0 ||
              messages[messages.length - 1].role !== 'user'
            ) {
              messages.push({
                role: 'user',
                content: text,
              });
            }
          }
        }
      }

      // Flush any remaining unanswered tool_calls at the end (safety net)
      for (const pending of pendingCalls) {
        messages.push({
          role: 'tool',
          tool_call_id: pending.id,
          content: 'Tool call was cancelled by the user.',
        });
      }

      // Final strict validation pass for DeepSeek/OpenAI schema:
      // Ensure NO message with role 'tool' exists unless the preceding message is an assistant with tool_calls.
      this.sanitizeToolCallSequences(messages);
    }

    const body: any = {
      model: this.resolveDeepSeekModel(request.model),
      messages,
      stream: false,
    };

    // Map Gemini tools to OpenAI function-calling format.
    // Tools are sorted alphabetically by name to ensure a deterministic
    // serialization across requests. DeepSeek's context cache matches by
    // exact prefix bytes, so any reordering of `tools[]` (which the Gemini
    // Core does not guarantee) would invalidate the cache prefix and
    // multiply input cost by ~10x for cache misses.
    const geminiTools = request.config?.tools;

    if (Array.isArray(geminiTools) && geminiTools.length > 0) {
      const openAiTools: any[] = [];
      for (const tool of geminiTools as any[]) {
        for (const decl of tool.functionDeclarations ?? []) {
          openAiTools.push({
            type: 'function',
            function: {
              name: decl.name,
              description: enrichToolDescription(
                decl.name,
                decl.description ?? '',
              ),
              parameters: decl.parameters ?? { type: 'object', properties: {} },
            },
          });
        }
      }
      if (openAiTools.length > 0) {
        openAiTools.sort((a: any, b: any) =>
          a.function.name.localeCompare(b.function.name),
        );
        body.tools = openAiTools;
      }
    }

    if (request.config?.responseMimeType === 'application/json') {
      body.response_format = { type: 'json_object' };
    }

    // Apply temperature and reasoning_effort correctly per model:
    // - V4-Flash (deepseek-chat): supports temperature AND reasoning_effort (thinking mode)
    // - V4-Pro (deepseek-reasoner): always-on deep reasoning, no extra params
    const resolvedModel = this.resolveDeepSeekModel(request.model);
    const isProModel =
      resolvedModel.includes('pro') || resolvedModel.includes('reasoner');

    if (!isProModel) {
      // Flash: send temperature (dynamic in real-time from config or request config)
      const currentTemp = this.config?.getTemperature?.() ?? this.temperature;
      const temp =
        request.config?.temperature !== undefined
          ? request.config.temperature
          : currentTemp;
      body.temperature = temp;

      // Flash: send reasoning_effort (dynamic in real-time from config)
      const currentEffort =
        this.config?.getReasoningEffort?.() ?? this.reasoningEffort;
      body.reasoning_effort = currentEffort;

      if (request.config?.topP !== undefined) {
        body.top_p = request.config.topP;
      }
    } else {
      // Pro: send pro reasoning_effort (dynamic in real-time from config: low, medium, high, max)
      const proEffort =
        (this.config as any)?.getProReasoningEffort?.() ?? 'high';
      body.reasoning_effort = proEffort;
      body.thinking = { type: 'enabled' };
    }

    if (request.config?.maxOutputTokens !== undefined) {
      body.max_tokens = request.config.maxOutputTokens;
    }

    // Thinking mode & reasoning effort configuration (DeepSeek V4-Pro & V4-Flash)
    const isReasoner =
      body.model.includes('reasoner') || body.model.includes('pro');
    const thinkingConfig = request.config?.thinkingConfig;
    const envReasoningEffort = process.env[
      'DEEPSEEK_REASONING_EFFORT'
    ]?.toLowerCase();
    const explicitEffort =
      (request.config as any)?.reasoning_effort ||
      (request.config as any)?.reasoningEffort;

    if (thinkingConfig?.thinkingBudget === 0) {
      body.thinking = { type: 'disabled' };
      delete body.reasoning_effort;
    } else {
      let effort: 'low' | 'high' | 'max' | undefined = undefined;

      if (
        explicitEffort &&
        ['low', 'high', 'max', 'medium', 'xhigh'].includes(explicitEffort)
      ) {
        effort =
          explicitEffort === 'low'
            ? 'low'
            : explicitEffort === 'max'
              ? 'max'
              : 'high';
      } else if (
        envReasoningEffort &&
        ['low', 'high', 'max'].includes(envReasoningEffort)
      ) {
        effort = envReasoningEffort as 'low' | 'high' | 'max';
      } else if (thinkingConfig?.thinkingLevel) {
        const levelStr = String(thinkingConfig.thinkingLevel).toLowerCase();
        if (levelStr.includes('low') || levelStr === 'minimal') {
          effort = 'low';
        } else if (levelStr.includes('max') || levelStr.includes('extreme')) {
          effort = 'max';
        } else {
          effort = 'high';
        }
      } else if (
        typeof thinkingConfig?.thinkingBudget === 'number' &&
        thinkingConfig.thinkingBudget > 0
      ) {
        if (thinkingConfig.thinkingBudget <= 2048) {
          effort = 'low';
        } else if (thinkingConfig.thinkingBudget >= 16384) {
          effort = 'max';
        } else {
          effort = 'high';
        }
      } else if (isReasoner) {
        effort = 'high';
      }

      if (effort) {
        body.reasoning_effort = effort;
        body.thinking = { type: 'enabled' };
      }
    }

    const pendingPrefix = (this.config as any)?.getAssistantPrefix?.();
    if (pendingPrefix) {
      body.messages.push({
        role: 'assistant',
        content: pendingPrefix,
        prefix: true,
      });
      (this.config as any)?.clearAssistantPrefix?.();
    }

    return body;
  }

  /**
   * Strictly enforces OpenAI / DeepSeek format for tool messages:
   * 1. Removes orphaned 'tool' messages that are not preceded by an assistant message with matching tool_calls id.
   * 2. Ensures every assistant tool_call has a following 'tool' response.
   */
  private sanitizeToolCallSequences(messages: any[]): void {
    let i = 0;
    while (i < messages.length) {
      const msg = messages[i];
      if (msg.role === 'tool') {
        let k = i - 1;
        while (k >= 0 && messages[k].role === 'tool') {
          k--;
        }
        const hasValidPrecedingCall =
          k >= 0 &&
          messages[k].role === 'assistant' &&
          Array.isArray(messages[k].tool_calls) &&
          messages[k].tool_calls.some((tc: any) => tc.id === msg.tool_call_id);

        if (!hasValidPrecedingCall) {
          debugLogger.warn(
            `[DeepSeek] Removing orphaned tool message (tool_call_id: ${msg.tool_call_id})`,
          );
          messages.splice(i, 1);
          continue;
        }
      }
      i++;
    }

    for (let idx = 0; idx < messages.length; idx++) {
      const msg = messages[idx];
      if (
        msg.role === 'assistant' &&
        Array.isArray(msg.tool_calls) &&
        msg.tool_calls.length > 0
      ) {
        const followingToolResponses: string[] = [];
        let nextIdx = idx + 1;
        while (nextIdx < messages.length && messages[nextIdx].role === 'tool') {
          followingToolResponses.push(messages[nextIdx].tool_call_id);
          nextIdx++;
        }

        for (const tc of msg.tool_calls) {
          if (!followingToolResponses.includes(tc.id)) {
            messages.splice(nextIdx, 0, {
              role: 'tool',
              tool_call_id: tc.id,
              content: 'Tool call was cancelled by the user.',
            });
            followingToolResponses.push(tc.id);
            nextIdx++;
          }
        }
      }
    }
  }

  private mapDeepSeekToGoogle(deepseekResponse: any): GenerateContentResponse {
    const choice = deepseekResponse.choices[0];
    const message = choice.message;
    const parts: any[] = [];

    // Map reasoning_content (thinking)
    if (message.reasoning_content) {
      parts.push({ text: message.reasoning_content, thought: true });
    } else if (message.content === null && !message.tool_calls) {
      // Safety net: if there is absolutely no content, add empty text part
      parts.push({ text: '' });
    }

    // Map tool_calls to functionCall parts
    if (message.tool_calls?.length > 0) {
      for (const tc of message.tool_calls) {
        let args = {};
        try {
          args = JSON.parse(tc.function.arguments);
        } catch {
          args = { _raw: tc.function.arguments };
        }
        parts.push({
          functionCall: { id: tc.id, name: tc.function.name, args },
        });
      }
    }

    // Map text content
    if (message.content) {
      parts.push({ text: message.content });
    }

    // Smuggle reasoning_content into a Part property so it survives the
    // Gemini Core's history pipeline (which strips `thought` parts and any
    // custom Content-level fields). Attach to the FIRST non-thought Part —
    // either a functionCall or a text part — both of which are preserved.
    // The thought part itself (if any) is filtered, so attaching to it would
    // be useless.
    if (message.reasoning_content) {
      const carrier = parts.find(
        (p: any) => p && !p.thought && (p.text !== undefined || p.functionCall),
      );
      if (carrier) {
        carrier[REASONING_FIELD] = message.reasoning_content;
      }
    }

    // Map DeepSeek's `prompt_cache_hit_tokens` to the standard
    // `cachedContentTokenCount` field so the existing telemetry, chat
    // recording service, and UI counters automatically reflect cache
    // efficiency. A high `cachedContentTokenCount / promptTokenCount` ratio
    // means the system is reusing prefix bytes at 1/10 the cost.
    const usage = deepseekResponse.usage ?? {};
    logCacheStats('non-stream', usage);
    const response: any = {
      candidates: [
        {
          content: {
            role: 'model',
            parts: parts.length > 0 ? parts : [{ text: '' }],
            // Best-effort: also save reasoning on the Content object. The
            // Gemini Core typically strips this, but it can help intra-request.
            reasoning_content: message.reasoning_content,
          },
          finishReason: 'STOP',
        },
      ],
      usageMetadata: {
        promptTokenCount: usage.prompt_tokens,
        candidatesTokenCount: usage.completion_tokens,
        totalTokenCount: usage.total_tokens,
        cachedContentTokenCount: usage.prompt_cache_hit_tokens,
      },
    };

    // Expose functionCalls for turn.ts compatibility (plain property, not class getter)
    const fnCalls = parts
      .filter((p: any) => p.functionCall)
      .map((p: any) => p.functionCall);
    if (fnCalls.length > 0) {
      response.functionCalls = fnCalls;
    }

    // Save to cache for the next turn using the stable signature
    if (message.reasoning_content) {
      const messageKey = this.getMessageKey(
        message.content || '',
        message.tool_calls,
      );
      debugAppend(
        `[CACHE SAVE] Saving for: ${(message.content || '').substring(0, 30)}... (Key: ${messageKey.substring(0, 50)})\n`,
      );
      DeepSeekContentGenerator.reasoningCache.set(
        messageKey,
        message.reasoning_content,
      );
      // Limit cache size
      if (DeepSeekContentGenerator.reasoningCache.size > 200) {
        const firstKey = DeepSeekContentGenerator.reasoningCache
          .keys()
          .next().value;
        if (firstKey !== undefined)
          DeepSeekContentGenerator.reasoningCache.delete(firstKey);
      }
      DeepSeekContentGenerator.saveCache();
    }

    return response as GenerateContentResponse;
  }

  async generateContent(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<GenerateContentResponse> {
    const body = this.mapGoogleToDeepSeek(request);

    const hasPrefix =
      Array.isArray(body.messages) &&
      body.messages.length > 0 &&
      body.messages[body.messages.length - 1]?.prefix === true;
    const endpointUrl = hasPrefix
      ? 'https://api.deepseek.com/beta/chat/completions'
      : `${this.baseUrl}/chat/completions`;

    debugLogger.debug(
      `[DeepSeek] Sending request to ${endpointUrl}`,
    );

    debugAppend(
      `--- REQUEST AT ${new Date().toISOString()} ---\n${JSON.stringify(body, null, 2)}\n\n`,
    );
    const userSignal =
      (request as any)?.abortSignal || (request as any)?.signal;
    const response = await fetch(endpointUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(body),
      signal: userSignal,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `DeepSeek API error: ${response.status} ${response.statusText} - ${errorText}`,
      );
    }

    const data = await response.json();
    return this.mapDeepSeekToGoogle(data);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const body = this.mapGoogleToDeepSeek(request);
    body.stream = true;
    // Required to receive a final usage chunk with `prompt_cache_hit_tokens`
    // and `prompt_cache_miss_tokens` — DeepSeek does not include usage in
    // streaming responses by default, which makes cost monitoring and cache
    // efficiency tracking impossible without this flag.
    body.stream_options = { include_usage: true };
    const self = this;

    const hasStreamPrefix =
      Array.isArray(body.messages) &&
      body.messages.length > 0 &&
      body.messages[body.messages.length - 1]?.prefix === true;
    const streamEndpointUrl = hasStreamPrefix
      ? 'https://api.deepseek.com/beta/chat/completions'
      : `${this.baseUrl}/chat/completions`;

    debugAppend(
      `--- STREAM REQUEST AT ${new Date().toISOString()} ---\n${JSON.stringify(body, null, 2)}\n\n`,
    );
    const streamUserSignal =
      (request as any)?.abortSignal || (request as any)?.signal;
    const response = await fetch(streamEndpointUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(body),
      signal: streamUserSignal,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `DeepSeek API error: ${response.status} ${response.statusText} - ${errorText}`,
      );
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('DeepSeek API error: Response body is null');
    }

    const decoder = new TextDecoder();

    async function* stream() {
      let buffer = '';
      const toolCallAcc: Record<
        number,
        { id: string; name: string; arguments: string }
      > = {};
      let hasToolCalls = false;
      let fullReasoning = '';
      let fullText = '';
      let latestUsage: any = undefined;

      while (true) {
        const { done, value } = await reader!.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed === 'data: [DONE]') continue;

          if (trimmed.startsWith('data: ')) {
            let json: any;
            try {
              json = JSON.parse(trimmed.substring(6));
            } catch {
              continue;
            }

            if (json.usage) {
              latestUsage = json.usage;
              logCacheStats('stream', json.usage);
            }

            const choice = json.choices?.[0];
            if (!choice) {
              // Yield usage metadata if delivered as standalone final chunk
              if (json.usage) {
                yield {
                  candidates: [],
                  usageMetadata: {
                    promptTokenCount: json.usage.prompt_tokens,
                    candidatesTokenCount: json.usage.completion_tokens,
                    totalTokenCount: json.usage.total_tokens,
                    cachedContentTokenCount: json.usage.prompt_cache_hit_tokens,
                  },
                } as unknown as GenerateContentResponse;
              }
              continue;
            }

            const delta = choice.delta ?? {};
            const deltaContent = delta.content;
            const deltaReasoning = delta.reasoning_content;
            const isFinished = choice.finish_reason != null;

            // Yield reasoning content chunks
            if (deltaReasoning) {
              fullReasoning += deltaReasoning;
              yield {
                candidates: [
                  {
                    content: {
                      role: 'model',
                      parts: [{ text: deltaReasoning, thought: true }],
                    },
                  },
                ],
              } as GenerateContentResponse;
            }

            // Accumulate streaming tool_calls deltas
            if (delta.tool_calls) {
              hasToolCalls = true;
              for (const tc of delta.tool_calls) {
                const idx: number = tc.index ?? 0;
                if (!toolCallAcc[idx]) {
                  toolCallAcc[idx] = {
                    id: tc.id ?? `call_${idx}`,
                    name: '',
                    arguments: '',
                  };
                }
                if (tc.id) toolCallAcc[idx].id = tc.id;
                if (tc.function?.name)
                  toolCallAcc[idx].name += tc.function.name;
                if (tc.function?.arguments)
                  toolCallAcc[idx].arguments += tc.function.arguments;
              }
            }

            // Yield text content chunks. The FIRST non-empty content delta
            // carries the accumulated `reasoning_content` as a smuggled
            // property on the Part. Because the Gemini Core consolidates
            // adjacent text parts into the FIRST one (mutating its `text`),
            // properties attached to that first part survive history
            // serialization. This gives us a reliable carrier for reasoning
            // across multi-turn conversations even when the disk cache fails.
            if (deltaContent) {
              const isFirstContentChunk = fullText.length === 0;
              fullText += deltaContent;
              const part: any = { text: deltaContent };
              if (isFirstContentChunk && fullReasoning) {
                part[REASONING_FIELD] = fullReasoning;
              }
              yield {
                candidates: [
                  {
                    content: { role: 'model', parts: [part] },
                  },
                ],
              } as GenerateContentResponse;
            }

            // Final chunk: yield tool calls or STOP finishReason
            if (isFinished) {
              const parts: any[] = [];
              const fnCalls: any[] = [];

              if (fullReasoning) {
                parts.push({ text: fullReasoning, thought: true });
              }

              if (hasToolCalls) {
                let firstFnCall = true;
                for (const idx of Object.keys(toolCallAcc).map(Number).sort()) {
                  const tc = toolCallAcc[idx];
                  let args = {};
                  try {
                    args = JSON.parse(tc.arguments);
                  } catch {
                    args = { _raw: tc.arguments };
                  }
                  const part: any = {
                    functionCall: { id: tc.id, name: tc.name, args },
                  };
                  // Smuggle reasoning_content on the FIRST functionCall part
                  // so it survives the Core's history pipeline even when the
                  // assistant produced no text content (the common case for
                  // tool-only sub-turns).
                  if (firstFnCall && fullReasoning) {
                    part[REASONING_FIELD] = fullReasoning;
                    firstFnCall = false;
                  }
                  parts.push(part);
                  fnCalls.push({ id: tc.id, name: tc.name, args });
                }
              }

              const usageToInclude = json.usage || latestUsage;
              const finalChunk: any = {
                candidates: [
                  {
                    content: {
                      role: 'model',
                      parts,
                      // Protection: Save the accumulated reasoning in the final object
                      reasoning_content: fullReasoning,
                    },
                    finishReason: 'STOP',
                  },
                ],
                ...(usageToInclude
                  ? {
                      usageMetadata: {
                        promptTokenCount: usageToInclude.prompt_tokens,
                        candidatesTokenCount: usageToInclude.completion_tokens,
                        totalTokenCount: usageToInclude.total_tokens,
                        cachedContentTokenCount:
                          usageToInclude.prompt_cache_hit_tokens,
                      },
                    }
                  : {}),
              };

              if (fnCalls.length > 0) {
                finalChunk.functionCalls = fnCalls;
              }

              // Save to cache for the next turn (Streaming) with stable signature
              if (fullReasoning) {
                // For streaming, reconstruct accumulated tool_calls if they exist
                const fnCallsForKey = hasToolCalls
                  ? Object.keys(toolCallAcc)
                      .map(Number)
                      .sort()
                      .map((idx) => ({
                        name: toolCallAcc[idx].name,
                        args: JSON.parse(toolCallAcc[idx].arguments || '{}'),
                      }))
                  : undefined;

                const messageKey = self.getMessageKey(fullText, fnCallsForKey);
                debugAppend(
                  `[CACHE SAVE STREAM] Saving for: ${fullText.substring(0, 30)}... (Key: ${messageKey.substring(0, 50)})\n`,
                );
                DeepSeekContentGenerator.reasoningCache.set(
                  messageKey,
                  fullReasoning,
                );
                if (DeepSeekContentGenerator.reasoningCache.size > 200) {
                  const firstKey = DeepSeekContentGenerator.reasoningCache
                    .keys()
                    .next().value;
                  if (firstKey !== undefined)
                    DeepSeekContentGenerator.reasoningCache.delete(firstKey);
                }
                DeepSeekContentGenerator.saveCache();
              }

              yield finalChunk as GenerateContentResponse;
            }
          }
        }
      }
    }

    return stream();
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    // DeepSeek doesn't have a dedicated token counting endpoint.
    // Estimate using the standard approximation: ~4 characters per token.
    // This is accurate enough for context window management.
    let charCount = 0;

    const contents = Array.isArray(request.contents)
      ? request.contents
      : request.contents
        ? [request.contents]
        : [];

    for (const content of contents as any[]) {
      for (const part of content.parts ?? []) {
        if (part.text) charCount += (part.text as string).length;
        if (part.functionCall)
          charCount += JSON.stringify(part.functionCall).length;
        if (part.functionResponse)
          charCount += JSON.stringify(part.functionResponse).length;
      }
    }

    // Also count system instruction if present
    const sysInstruction = (request as any).config?.systemInstruction;
    if (sysInstruction) {
      charCount += JSON.stringify(sysInstruction).length;
    }

    const totalTokens = Math.ceil(charCount / 4);
    return { totalTokens };
  }

  async embedContent(
    _request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    // DeepSeek doesn't natively support embeddings.
    // Return a zero-vector so features that call embedContent don't crash.
    debugLogger.warn(
      '[DeepSeek] embedContent is not supported — returning zero vector.',
    );
    return {
      embedding: { values: new Array(256).fill(0) },
    } as unknown as EmbedContentResponse;
  }
}
