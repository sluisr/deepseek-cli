/**
 * @license
 * Copyright 2025 Google LLC
 * Copyright 2025 sluisr (DeepSeek adaptation)
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi } from 'vitest';
import { DeepSeekContentGenerator } from './deepseekContentGenerator.js';
import type { GenerateContentParameters } from '@google/genai';
import { convertSessionToClientHistory } from '../utils/sessionUtils.js';
import { CoreToolCallStatus } from '../scheduler/types.js';

describe('DeepSeekContentGenerator', () => {
  const generator = new DeepSeekContentGenerator('test-api-key');

  it('maps simple user message correctly', () => {
    const request: GenerateContentParameters = {
      model: 'deepseek-v4-flash',
      contents: [
        {
          role: 'user',
          parts: [{ text: 'hola' }],
        },
      ],
    };

    // Access private method for testing message transformation
    const body = (generator as any).mapGoogleToDeepSeek(request);

    expect(body.model).toBe('deepseek-v4-flash');
    expect(body.messages.some((m: any) => m.role === 'user' && m.content === 'hola')).toBe(true);
  });

  it('correctly serializes function calls and matching function responses', () => {
    const request: GenerateContentParameters = {
      model: 'deepseek-v4-flash',
      contents: [
        {
          role: 'user',
          parts: [{ text: 'List files' }],
        },
        {
          role: 'model',
          parts: [
            {
              functionCall: {
                id: 'call_abc_123',
                name: 'list_directory',
                args: { dir: '.' },
              },
            },
          ],
        },
        {
          role: 'user',
          parts: [
            {
              functionResponse: {
                id: 'call_abc_123',
                name: 'list_directory',
                response: { output: 'file1.txt\nfile2.txt' },
              },
            },
          ],
        },
      ],
    };

    const body = (generator as any).mapGoogleToDeepSeek(request);
    const messages = body.messages;

    const assistantMsg = messages.find((m: any) => m.role === 'assistant');
    expect(assistantMsg).toBeDefined();
    expect(assistantMsg.tool_calls).toHaveLength(1);
    expect(assistantMsg.tool_calls[0].id).toBe('call_abc_123');

    const toolMsg = messages.find((m: any) => m.role === 'tool');
    expect(toolMsg).toBeDefined();
    expect(toolMsg.tool_call_id).toBe('call_abc_123');
    expect(toolMsg.content).toContain('file1.txt');
  });

  it('skips unmatched or orphaned function responses without emitting illegal role: "tool"', () => {
    const request: GenerateContentParameters = {
      model: 'deepseek-v4-flash',
      contents: [
        {
          role: 'user',
          parts: [{ text: 'hola' }],
        },
        {
          role: 'user',
          parts: [
            {
              functionResponse: {
                id: 'call_orphaned',
                name: 'unknown_tool',
                response: { output: 'some result' },
              },
            },
            { text: 'follow up question' },
          ],
        },
      ],
    };

    const body = (generator as any).mapGoogleToDeepSeek(request);
    const messages = body.messages;

    // Must NOT contain any orphaned role: 'tool' because there was no preceding assistant tool_calls
    const toolMessages = messages.filter((m: any) => m.role === 'tool');
    expect(toolMessages).toHaveLength(0);

    // Should contain the valid user text
    expect(messages.some((m: any) => m.role === 'user' && m.content.includes('follow up'))).toBe(true);
  });

  it('handles resumed session with tool calls and subsequent user message without error', () => {
    // Simulate messages loaded from session recording
    const recordedMessages: any[] = [
      {
        id: 'msg1',
        type: 'user',
        content: 'Check status',
      },
      {
        id: 'msg2',
        type: 'gemini',
        content: '',
        toolCalls: [
          {
            id: 'call_status_1',
            name: 'git_status',
            args: {},
            status: CoreToolCallStatus.Success,
            result: 'On branch main',
          },
        ],
      },
      {
        id: 'msg3',
        type: 'user',
        content: [
          {
            functionResponse: {
              id: 'call_status_1',
              name: 'git_status',
              response: { output: 'On branch main' },
            },
          },
        ],
      },
    ];

    const history = convertSessionToClientHistory(recordedMessages);

    // User resumes session and sends "hola"
    const request: GenerateContentParameters = {
      model: 'deepseek-v4-flash',
      contents: [
        ...history.map((h) => h.content),
        {
          role: 'user',
          parts: [{ text: 'hola' }],
        },
      ],
    };

    const body = (generator as any).mapGoogleToDeepSeek(request);
    const messages = body.messages;

    // Verify all tool messages are strictly preceded by an assistant message with matching tool_calls
    for (let i = 0; i < messages.length; i++) {
      if (messages[i].role === 'tool') {
        let k = i - 1;
        while (k >= 0 && messages[k].role === 'tool') k--;
        expect(k).toBeGreaterThanOrEqual(0);
        expect(messages[k].role).toBe('assistant');
        expect(
          messages[k].tool_calls.some(
            (tc: any) => tc.id === messages[i].tool_call_id,
          ),
        ).toBe(true);
      }
    }

    // Verify the latest user message is present
    expect(messages[messages.length - 1]).toEqual({
      role: 'user',
      content: 'hola',
    });
  });

  it('synthesizes cancellation responses for interrupted assistant tool calls before new user turn', () => {
    const request: GenerateContentParameters = {
      model: 'deepseek-v4-flash',
      contents: [
        {
          role: 'user',
          parts: [{ text: 'Run command' }],
        },
        {
          role: 'model',
          parts: [
            {
              functionCall: {
                id: 'call_interrupted',
                name: 'long_task',
                args: {},
              },
            },
          ],
        },
        {
          role: 'user',
          parts: [{ text: 'cancel and do something else' }],
        },
      ],
    };

    const body = (generator as any).mapGoogleToDeepSeek(request);
    const messages = body.messages;

    const assistantIdx = messages.findIndex(
      (m: any) =>
        m.role === 'assistant' &&
        m.tool_calls?.some((tc: any) => tc.id === 'call_interrupted'),
    );
    expect(assistantIdx).toBeGreaterThanOrEqual(0);

    // Immediately after assistant must come the tool cancellation response
    expect(messages[assistantIdx + 1]).toEqual({
      role: 'tool',
      tool_call_id: 'call_interrupted',
      content: 'Tool call was cancelled by the user.',
    });

    // Followed by the new user message
    expect(messages[assistantIdx + 2]).toEqual({
      role: 'user',
      content: 'cancel and do something else',
    });
  });

  it('estimates token count accurately without errors', async () => {
    const count = await generator.countTokens({
      contents: [
        {
          role: 'user',
          parts: [{ text: 'Hello, how are you today?' }],
        },
      ],
    });

    expect(count.totalTokens).toBeGreaterThan(0);
  });

  it('configures thinking and reasoning_effort correctly for pro and flash models', () => {
    // 1. Pro model defaults to thinking enabled with high effort
    const proBody = (generator as any).mapGoogleToDeepSeek({
      model: 'deepseek-v4-pro',
      contents: [{ role: 'user', parts: [{ text: 'Solve math problem' }] }],
    });
    expect(proBody.thinking).toEqual({ type: 'enabled' });
    expect(proBody.reasoning_effort).toBe('high');

    // 2. Flash model with thinkingLevel LOW maps to low effort
    const flashLowBody = (generator as any).mapGoogleToDeepSeek({
      model: 'deepseek-v4-flash',
      contents: [{ role: 'user', parts: [{ text: 'Quick summary' }] }],
      config: {
        thinkingConfig: {
          thinkingLevel: 'LOW',
        },
      },
    });
    expect(flashLowBody.thinking).toEqual({ type: 'enabled' });
    expect(flashLowBody.reasoning_effort).toBe('low');

    // 3. Explicit reasoning_effort: max
    const maxBody = (generator as any).mapGoogleToDeepSeek({
      model: 'deepseek-v4-pro',
      contents: [{ role: 'user', parts: [{ text: 'Complex architecture' }] }],
      config: {
        reasoning_effort: 'max',
      } as any,
    });
    expect(maxBody.thinking).toEqual({ type: 'enabled' });
    expect(maxBody.reasoning_effort).toBe('max');

    // 4. thinkingBudget = 0 disables thinking mode
    const disabledBody = (generator as any).mapGoogleToDeepSeek({
      model: 'deepseek-v4-flash',
      contents: [{ role: 'user', parts: [{ text: 'Fast route' }] }],
      config: {
        thinkingConfig: {
          thinkingBudget: 0,
        },
      },
    });
    expect(disabledBody.thinking).toEqual({ type: 'disabled' });
    expect(disabledBody.reasoning_effort).toBeUndefined();
  });

  it('attaches prefix message when getAssistantPrefix is provided by config', () => {
    const mockConfig = {
      getAssistantPrefix: vi.fn().mockReturnValue('```json\n{'),
      clearAssistantPrefix: vi.fn(),
    };
    const prefixGen = new DeepSeekContentGenerator(
      'test-key',
      'https://api.deepseek.com',
      mockConfig as any,
    );

    const body = (prefixGen as any).mapGoogleToDeepSeek({
      model: 'deepseek-v4-flash',
      contents: [{ role: 'user', parts: [{ text: 'Give me JSON' }] }],
    });

    const lastMsg = body.messages[body.messages.length - 1];
    expect(lastMsg).toEqual({
      role: 'assistant',
      content: '```json\n{',
      prefix: true,
    });
    expect(mockConfig.clearAssistantPrefix).toHaveBeenCalled();
  });
});

