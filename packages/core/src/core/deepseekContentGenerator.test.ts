/**
 * @license
 * Copyright 2026 DeepSeek CLI Authors
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { DeepSeekContentGenerator } from './deepseekContentGenerator.js';
import type { GenerateContentParameters } from '@google/genai';
import { LlmRole } from '../telemetry/llmRole.js';

describe('DeepSeekContentGenerator', () => {
  let generator: DeepSeekContentGenerator;
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    generator = new DeepSeekContentGenerator('test-api-key', 'https://api.deepseek.com');
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  describe('mapGoogleToDeepSeek (Tool & History mapping)', () => {
    it('correctly maps tool calls and matching tool responses even with tool name prefixes', () => {
      const request: GenerateContentParameters = {
        model: 'deepseek-v4-flash',
        contents: [
          {
            role: 'user',
            parts: [{ text: 'Please read foo.ts' }],
          },
          {
            role: 'model',
            parts: [
              {
                functionCall: {
                  id: 'call_00_12345',
                  name: 'read_file',
                  args: { file_path: 'foo.ts' },
                },
              },
            ],
          },
          {
            role: 'user',
            parts: [
              {
                functionResponse: {
                  id: 'read_file__call_00_12345',
                  name: 'read_file',
                  response: { content: 'console.log("hello");' },
                },
              },
            ],
          },
        ],
      };

      const body = (generator as any).mapGoogleToDeepSeek(request);
      const conversationMessages = body.messages.filter((m: any) => m.role !== 'system');
      expect(conversationMessages).toHaveLength(3);
      expect(conversationMessages[0]).toEqual({ role: 'user', content: 'Please read foo.ts' });
      expect(conversationMessages[1].role).toBe('assistant');
      expect(conversationMessages[1].tool_calls).toHaveLength(1);
      expect(conversationMessages[1].tool_calls[0].id).toBe('call_00_12345');

      expect(conversationMessages[2].role).toBe('tool');
      expect(conversationMessages[2].tool_call_id).toBe('call_00_12345');
      expect(conversationMessages[2].content).toBe('{"content":"console.log(\\"hello\\");"}');
    });

    it('preserves pending tool calls across non-tool intermediate user turns', () => {
      const request: GenerateContentParameters = {
        model: 'deepseek-v4-flash',
        contents: [
          {
            role: 'user',
            parts: [{ text: 'Please read bar.ts' }],
          },
          {
            role: 'model',
            parts: [
              {
                functionCall: {
                  id: 'call_bar_01',
                  name: 'read_file',
                  args: { file_path: 'bar.ts' },
                },
              },
            ],
          },
          {
            role: 'user',
            parts: [{ text: 'Here is user editor context update' }],
          },
          {
            role: 'user',
            parts: [
              {
                functionResponse: {
                  id: 'call_bar_01',
                  name: 'read_file',
                  response: { content: 'export const x = 1;' },
                },
              },
            ],
          },
        ],
      };

      const body = (generator as any).mapGoogleToDeepSeek(request);
      const toolMessages = body.messages.filter((m: any) => m.role === 'tool');
      expect(toolMessages).toHaveLength(1);
      expect(toolMessages[0].tool_call_id).toBe('call_bar_01');
      expect(toolMessages[0].content).toBe('{"content":"export const x = 1;"}');
    });
  });

  describe('generateContentStream', () => {
    it('yields a final chunk with finishReason STOP even if SSE stream closes without finish_reason delta', async () => {
      const sseData = [
        'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n',
        'data: {"choices":[{"delta":{"content":"world!"}}]}\n\n',
        'data: [DONE]\n\n',
      ].join('');

      const streamResponse = new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(sseData));
          controller.close();
        },
      });

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        body: streamResponse,
      } as unknown as Response);

      const request: GenerateContentParameters = {
        model: 'deepseek-v4-flash',
        contents: [{ role: 'user', parts: [{ text: 'Hi' }] }],
      };

      const genStream = await generator.generateContentStream(request, 'prompt-1', LlmRole.MAIN);
      const chunks = [];
      for await (const chunk of genStream) {
        chunks.push(chunk);
      }

      // Must have received text chunks and a final chunk with finishReason STOP
      const finishChunk = chunks.find((c) => c.candidates?.[0]?.finishReason === 'STOP');
      expect(finishChunk).toBeDefined();
    });

    it('retries on HTTP 429 and 503 before succeeding', async () => {
      let callCount = 0;
      const sseData = 'data: {"choices":[{"delta":{"content":"Success after retry"}}]}\n\ndata: [DONE]\n\n';

      globalThis.fetch = vi.fn().mockImplementation(async () => {
        callCount++;
        if (callCount === 1) {
          return {
            ok: false,
            status: 429,
            statusText: 'Too Many Requests',
            text: async () => 'Rate limit exceeded',
          } as Response;
        }
        return {
          ok: true,
          status: 200,
          body: new ReadableStream({
            start(controller) {
              controller.enqueue(new TextEncoder().encode(sseData));
              controller.close();
            },
          }),
        } as unknown as Response;
      });

      const request: GenerateContentParameters = {
        model: 'deepseek-v4-flash',
        contents: [{ role: 'user', parts: [{ text: 'Hi' }] }],
      };

      const genStream = await generator.generateContentStream(request, 'prompt-2', LlmRole.MAIN);
      const chunks = [];
      for await (const chunk of genStream) {
        chunks.push(chunk);
      }

      expect(callCount).toBe(2);
      expect(chunks.length).toBeGreaterThan(0);
    });
  });

  describe('generateContent', () => {
    it('retries on HTTP 503 before succeeding with JSON response', async () => {
      let callCount = 0;
      globalThis.fetch = vi.fn().mockImplementation(async () => {
        callCount++;
        if (callCount === 1) {
          return {
            ok: false,
            status: 503,
            statusText: 'Service Unavailable',
            text: async () => 'Server overloaded',
          } as Response;
        }
        return {
          ok: true,
          status: 200,
          json: async () => ({
            choices: [
              {
                message: {
                  role: 'assistant',
                  content: 'Hello world!',
                },
                finish_reason: 'stop',
              },
            ],
            usage: {
              prompt_tokens: 10,
              completion_tokens: 5,
              total_tokens: 15,
            },
          }),
        } as unknown as Response;
      });

      const request: GenerateContentParameters = {
        model: 'deepseek-v4-flash',
        contents: [{ role: 'user', parts: [{ text: 'Hello' }] }],
      };

      const res = await generator.generateContent(request, 'prompt-3', LlmRole.MAIN);
      expect(callCount).toBe(2);
      expect(res.candidates?.[0]?.content?.parts?.[0]?.text).toBe('Hello world!');
    });
  });
});
