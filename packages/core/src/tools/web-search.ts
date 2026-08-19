/**
 * @license
 * Copyright 2025 Google LLC
 * Copyright 2026 sluisr
 * SPDX-License-Identifier: Apache-2.0
 */

import type { MessageBus } from '../confirmation-bus/message-bus.js';
import { WEB_SEARCH_TOOL_NAME, WEB_SEARCH_DISPLAY_NAME } from './tool-names.js';
import type { GroundingMetadata } from '@google/genai';
import {
  BaseDeclarativeTool,
  BaseToolInvocation,
  Kind,
  type ToolInvocation,
  type ToolResult,
  type ExecuteOptions,
} from './tools.js';
import { ToolErrorType } from './tool-error.js';

import { getErrorMessage, isAbortError } from '../utils/errors.js';
import { getResponseText } from '../utils/partUtils.js';
import { debugLogger } from '../utils/debugLogger.js';
import { WEB_SEARCH_DEFINITION } from './definitions/coreTools.js';
import { resolveToolDeclaration } from './definitions/resolver.js';
import { LlmRole } from '../telemetry/llmRole.js';
import type { AgentLoopContext } from '../config/agent-loop-context.js';
import { AuthType } from '../core/contentGenerator.js';
import { loadDeepSeekApiKey } from '../core/deepseekApiKeyStorage.js';

interface GroundingChunkWeb {
  uri?: string;
  title?: string;
}

interface GroundingChunkItem {
  web?: GroundingChunkWeb;
  // Other properties might exist if needed in the future
}

interface GroundingSupportSegment {
  startIndex: number;
  endIndex: number;
  text?: string; // text is optional as per the example
}

interface GroundingSupportItem {
  segment?: GroundingSupportSegment;
  groundingChunkIndices?: number[];
  confidenceScores?: number[]; // Optional as per example
}

/**
 * Parameters for the WebSearchTool.
 */
export interface WebSearchToolParams {
  /**
   * The search query.
   */

  query: string;
}

/**
 * Extends ToolResult to include sources for web search.
 */
export interface WebSearchToolResult extends ToolResult {
  sources?: GroundingMetadata extends { groundingChunks: GroundingChunkItem[] }
    ? GroundingMetadata['groundingChunks']
    : GroundingChunkItem[];
}

class WebSearchToolInvocation extends BaseToolInvocation<
  WebSearchToolParams,
  WebSearchToolResult
> {
  constructor(
    private readonly context: AgentLoopContext,
    params: WebSearchToolParams,
    messageBus: MessageBus,
    _toolName?: string,
    _toolDisplayName?: string,
  ) {
    super(params, messageBus, _toolName, _toolDisplayName);
  }

  override getDescription(): string {
    return `Searching the web for: "${this.params.query}"`;
  }

  async execute({
    abortSignal: signal,
  }: ExecuteOptions): Promise<WebSearchToolResult> {
    const config = this.context.config;
    const isDeepSeekAuth =
      config?.getContentGeneratorConfig?.()?.authType === AuthType.USE_DEEPSEEK;

    // When running on DeepSeek API: use native DeepSeek Responses web_search engine
    if (isDeepSeekAuth) {
      const query = this.params.query;
      const apiKey =
        config?.getContentGeneratorConfig?.()?.apiKey ||
        process.env['DEEPSEEK_API_KEY'] ||
        (await loadDeepSeekApiKey());

      if (apiKey) {
        try {
          const currentModel = config?.getModel?.() || '';
          const isPro =
            currentModel.includes('pro') || currentModel.includes('reasoner');
          const searchEffort: string = isPro
            ? ((config as any)?.getProSearchReasoningEffort?.() ?? 'high')
            : (config?.getSearchReasoningEffort?.() ?? 'low');
          const searchModel = isPro ? 'deepseek-reasoner' : 'deepseek-chat';

          const maxTokens =
            searchEffort === 'max'
              ? 6000
              : searchEffort === 'high'
                ? 4000
                : searchEffort === 'medium'
                  ? 2000
                  : 800;

          const controller = new AbortController();
          const timeoutMs =
            searchEffort === 'max'
              ? 25000
              : searchEffort === 'high'
                ? 18000
                : searchEffort === 'medium'
                  ? 10000
                  : 5000;
          const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
          const combinedSignal = signal
            ? AbortSignal.any([signal, controller.signal])
            : controller.signal;

          const res = await fetch('https://api.deepseek.com/responses', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${apiKey}`,
            },
            body: JSON.stringify({
              model: searchModel,
              instructions:
                searchEffort === 'max' || searchEffort === 'high'
                  ? 'Search the web thoroughly and synthesize all key facts, full context, and exact source links.'
                  : searchEffort === 'medium'
                    ? 'Search the web and return a well-structured summary with key facts and source URLs.'
                    : 'Search the web and return concise factual results with URLs and relevant information quickly.',
              input: query,
              reasoning: { effort: searchEffort === 'max' ? 'high' : searchEffort },
              max_output_tokens: maxTokens,
              tools: [{ type: 'web_search' }],
            }),
            signal: combinedSignal,
          });
          clearTimeout(timeoutId);

          if (res.ok) {
            const data: any = await res.json();
            let textContent = '';
            let finalAnswer = '';
            for (const item of data.output || []) {
              if (item.type === 'message' && Array.isArray(item.content)) {
                for (const c of item.content) {
                  if (c.type === 'output_text' && c.text) {
                    if (item.phase === 'final_answer') {
                      finalAnswer += (finalAnswer ? '\n\n' : '') + c.text;
                    } else {
                      textContent += c.text + '\n\n';
                    }
                  }
                }
              }
            }

            const resultText = (finalAnswer || textContent).trim();
            if (resultText) {
              return {
                llmContent: `DeepSeek Native WebSearch results for "${query}":\n\n${resultText}`,
                returnDisplay: `DeepSeek WebSearch completed for "${query}".`,
              };
            }
          }
        } catch (err: any) {
          debugLogger.warn(
            'DeepSeek native /responses web_search failed, using fast fallback',
            err,
          );
        }
      }

      // Fast fallback search
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 6000);
        const combinedSignal = signal
          ? AbortSignal.any([signal, controller.signal])
          : controller.signal;

        const fetchHtml = fetch('https://html.duckduckgo.com/html/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent':
              'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            Accept: 'text/html',
          },
          body: `q=${encodeURIComponent(query)}`,
          signal: combinedSignal,
        });

        const fetchLite = fetch('https://lite.duckduckgo.com/lite/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent':
              'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0',
            Accept: 'text/html',
          },
          body: `q=${encodeURIComponent(query)}`,
          signal: combinedSignal,
        });

        const res = await Promise.any([
          fetchHtml.then((r) => (r.ok ? r : Promise.reject(r))),
          fetchLite.then((r) => (r.ok ? r : Promise.reject(r))),
        ]);
        clearTimeout(timeoutId);

        const html = await res.text();
        const results: Array<{ title: string; snippet: string; url: string }> =
          [];

        const linkRegex =
          /<a[^>]*class=['"](?:result-link|result__url|result__snippet|result__a)['"][^>]*href=['"]([^'"]+)['"][^>]*>([\s\S]*?)<\/a>/gi;
        const snippetRegex =
          /<(?:td|a|div)[^>]*class=['"](?:result-snippet|result__snippet)['"][^>]*>([\s\S]*?)<\/(?:td|a|div)>/gi;

        const links: Array<{ url: string; title: string }> = [];
        let m: RegExpExecArray | null;
        while ((m = linkRegex.exec(html)) !== null) {
          let cleanUrl = m[1].replace(/&amp;/g, '&');
          if (cleanUrl.includes('uddg=')) {
            const match = cleanUrl.match(/uddg=([^&]+)/);
            if (match) cleanUrl = decodeURIComponent(match[1]);
          }
          if (cleanUrl.startsWith('http')) {
            links.push({
              url: cleanUrl,
              title: m[2].replace(/<[^>]+>/g, '').trim(),
            });
          }
        }

        const snippets: string[] = [];
        while ((m = snippetRegex.exec(html)) !== null) {
          snippets.push(
            m[1]
              .replace(/<[^>]+>/g, '')
              .replace(/\s+/g, ' ')
              .trim(),
          );
        }

        for (let i = 0; i < Math.min(links.length, 6); i++) {
          if (links[i].url && links[i].title) {
            results.push({
              title: links[i].title,
              url: links[i].url,
              snippet: snippets[i] || '',
            });
          }
        }

        if (results.length > 0) {
          const formatted = results
            .map(
              (r, i) =>
                `[${i + 1}] ${r.title}\nURL: ${r.url}\nSummary: ${r.snippet}`,
            )
            .join('\n\n');

          return {
            llmContent: `Web search results for "${query}":\n\n${formatted}`,
            returnDisplay: `Found ${results.length} web results for "${query}".`,
          };
        }
      } catch (err: unknown) {
        if (isAbortError(err)) {
          return {
            llmContent: 'Web search was cancelled.',
            returnDisplay: 'Search cancelled.',
          };
        }
        debugLogger.warn('Fast web search fallback failed:', err);
      }

      return {
        llmContent: `No search results found on the web for query: "${query}"`,
        returnDisplay: `No web results for "${query}".`,
      };
    }

    const geminiClient = this.context.geminiClient;

    try {
      const response = await geminiClient.generateContent(
        { model: 'web-search' },
        [{ role: 'user', parts: [{ text: this.params.query }] }],
        signal,
        LlmRole.UTILITY_TOOL,
      );

      const responseText = getResponseText(response);
      const groundingMetadata = response.candidates?.[0]?.groundingMetadata;
      const sources = groundingMetadata?.groundingChunks as
        | GroundingChunkItem[]
        | undefined;
      // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
      const groundingSupports = groundingMetadata?.groundingSupports as
        | GroundingSupportItem[]
        | undefined;

      if (!responseText || !responseText.trim()) {
        return {
          llmContent: `No search results or information found for query: "${this.params.query}"`,
          returnDisplay: 'No information found.',
        };
      }

      let modifiedResponseText = responseText;
      const sourceListFormatted: string[] = [];

      if (sources && sources.length > 0) {
        sources.forEach((source: GroundingChunkItem, index: number) => {
          const title = source.web?.title || 'Untitled';
          const uri = source.web?.uri || 'No URI';
          sourceListFormatted.push(`[${index + 1}] ${title} (${uri})`);
        });

        if (groundingSupports && groundingSupports.length > 0) {
          const insertions: Array<{ index: number; marker: string }> = [];
          groundingSupports.forEach((support: GroundingSupportItem) => {
            if (support.segment && support.groundingChunkIndices) {
              const citationMarker = support.groundingChunkIndices
                .map((chunkIndex: number) => `[${chunkIndex + 1}]`)
                .join('');
              insertions.push({
                index: support.segment.endIndex,
                marker: citationMarker,
              });
            }
          });

          // Sort insertions by index in descending order to avoid shifting subsequent indices
          insertions.sort((a, b) => b.index - a.index);

          // Use TextEncoder/TextDecoder since segment indices are UTF-8 byte positions
          const encoder = new TextEncoder();
          const responseBytes = encoder.encode(modifiedResponseText);
          const parts: Uint8Array[] = [];
          let lastIndex = responseBytes.length;
          for (const ins of insertions) {
            const pos = Math.min(ins.index, lastIndex);
            parts.unshift(responseBytes.subarray(pos, lastIndex));
            parts.unshift(encoder.encode(ins.marker));
            lastIndex = pos;
          }
          parts.unshift(responseBytes.subarray(0, lastIndex));

          // Concatenate all parts into a single buffer
          const totalLength = parts.reduce((sum, part) => sum + part.length, 0);
          const finalBytes = new Uint8Array(totalLength);
          let offset = 0;
          for (const part of parts) {
            finalBytes.set(part, offset);
            offset += part.length;
          }
          modifiedResponseText = new TextDecoder().decode(finalBytes);
        }

        if (sourceListFormatted.length > 0) {
          modifiedResponseText +=
            '\n\nSources:\n' + sourceListFormatted.join('\n');
        }
      }

      return {
        llmContent: `Web search results for "${this.params.query}":\n\n${modifiedResponseText}`,
        returnDisplay: `Search results for "${this.params.query}" returned.`,
        sources,
      };
    } catch (error: unknown) {
      if (isAbortError(error)) {
        return {
          llmContent: 'Web search was cancelled.',
          returnDisplay: 'Search cancelled.',
        };
      }
      const errorMessage = `Error during web search for query "${
        this.params.query
      }": ${getErrorMessage(error)}`;
      debugLogger.warn(errorMessage, error);
      return {
        llmContent: `Error: ${errorMessage}`,
        returnDisplay: `Error performing web search.`,
        error: {
          message: errorMessage,
          type: ToolErrorType.WEB_SEARCH_FAILED,
        },
      };
    }
  }
}

/**
 * A tool to perform web searches using Google Search via the Gemini API.
 */
export class WebSearchTool extends BaseDeclarativeTool<
  WebSearchToolParams,
  WebSearchToolResult
> {
  static readonly Name = WEB_SEARCH_TOOL_NAME;

  constructor(
    private readonly context: AgentLoopContext,
    messageBus: MessageBus,
  ) {
    super(
      WebSearchTool.Name,
      WEB_SEARCH_DISPLAY_NAME,
      WEB_SEARCH_DEFINITION.base.description!,
      Kind.Search,
      WEB_SEARCH_DEFINITION.base.parametersJsonSchema,
      messageBus,
      true, // isOutputMarkdown
      false, // canUpdateOutput
    );
  }

  /**
   * Validates the parameters for the WebSearchTool.
   * @param params The parameters to validate
   * @returns An error message string if validation fails, null if valid
   */
  protected override validateToolParamValues(
    params: WebSearchToolParams,
  ): string | null {
    if (!params.query || params.query.trim() === '') {
      return "The 'query' parameter cannot be empty.";
    }
    return null;
  }

  protected createInvocation(
    params: WebSearchToolParams,
    messageBus: MessageBus,
    _toolName?: string,
    _toolDisplayName?: string,
  ): ToolInvocation<WebSearchToolParams, WebSearchToolResult> {
    return new WebSearchToolInvocation(
      this.context.config,
      params,
      messageBus ?? this.messageBus,
      _toolName,
      _toolDisplayName,
    );
  }

  override getSchema(modelId?: string) {
    return resolveToolDeclaration(WEB_SEARCH_DEFINITION, modelId);
  }
}
