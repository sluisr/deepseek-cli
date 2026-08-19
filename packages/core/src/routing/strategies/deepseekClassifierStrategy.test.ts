/**
 * @license
 * Copyright 2026 DeepSeek CLI Authors
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { DeepSeekClassifierStrategy } from './deepseekClassifierStrategy.js';
import type { RoutingContext } from '../routingStrategy.js';
import type { BaseLlmClient } from '../../core/baseLlmClient.js';
import type { Config } from '../../config/config.js';
import { DEEPSEEK_CHAT_MODEL, DEEPSEEK_REASONER_MODEL } from '../../config/models.js';
import type { LocalLiteRtLmClient } from '../../core/localLiteRtLmClient.js';

vi.mock('../../core/deepseekApiKeyStorage.js', () => ({
  loadDeepSeekApiKey: vi.fn().mockResolvedValue('test-deepseek-key'),
}));

describe('DeepSeekClassifierStrategy', () => {
  const strategy = new DeepSeekClassifierStrategy();
  const mockClient = {} as BaseLlmClient;
  const mockLocalLiteRtLmClient = {} as LocalLiteRtLmClient;

  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
    process.env['DEEPSEEK_API_KEY'] = 'test-deepseek-key';
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it('should return null when the model is not a deepseek model', async () => {
    const mockConfig = {
      getModel: () => 'gemini-2.5-flash',
    } as Config;

    const mockContext: RoutingContext = {
      history: [],
      request: 'Hello',
      signal: new AbortController().signal,
    };

    const decision = await strategy.route(
      mockContext,
      mockConfig,
      mockClient,
      mockLocalLiteRtLmClient,
    );
    expect(decision).toBeNull();
  });

  it('should return null when the model is already deepseek-v4-pro / reasoner', async () => {
    const mockConfig = {
      getModel: () => DEEPSEEK_REASONER_MODEL,
    } as Config;

    const mockContext: RoutingContext = {
      history: [],
      request: 'Refactor the entire authentication architecture',
      signal: new AbortController().signal,
    };

    const decision = await strategy.route(
      mockContext,
      mockConfig,
      mockClient,
      mockLocalLiteRtLmClient,
    );
    expect(decision).toBeNull();
  });

  it('should fast-path trivial confirmations without making network calls', async () => {
    const mockConfig = {
      getModel: () => DEEPSEEK_CHAT_MODEL,
    } as Config;

    const fetchSpy = vi.fn();
    globalThis.fetch = fetchSpy;

    const trivialInputs = ['aslo', 'hazlo', 'ok', 'si', 'sí', 'yes', 'sigo', 'dale'];
    for (const input of trivialInputs) {
      const mockContext: RoutingContext = {
        history: [],
        request: input,
        signal: new AbortController().signal,
      };

      const decision = await strategy.route(
        mockContext,
        mockConfig,
        mockClient,
        mockLocalLiteRtLmClient,
      );

      expect(decision).not.toBeNull();
      expect(decision?.model).toBe(DEEPSEEK_CHAT_MODEL);
      expect(decision?.metadata.latencyMs).toBe(0);
      expect(decision?.metadata.reasoning).toContain('[FastPath]');
    }

    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('should fast-path empty or non-text requests to null without network calls', async () => {
    const mockConfig = {
      getModel: () => DEEPSEEK_CHAT_MODEL,
    } as Config;

    const fetchSpy = vi.fn();
    globalThis.fetch = fetchSpy;

    const mockContext: RoutingContext = {
      history: [],
      request: [],
      signal: new AbortController().signal,
    };

    const decision = await strategy.route(
      mockContext,
      mockConfig,
      mockClient,
      mockLocalLiteRtLmClient,
    );

    expect(decision).toBeNull();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('should escalate to deepseek-v4-pro when score >= 70', async () => {
    const mockConfig = {
      getModel: () => DEEPSEEK_CHAT_MODEL,
    } as Config;

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [
          {
            message: {
              content: JSON.stringify({
                score: 85,
                reason: 'Complex architecture refactor across multiple modules',
              }),
            },
          },
        ],
      }),
    } as Response);

    const mockContext: RoutingContext = {
      history: [],
      request: 'Redesign the compiler pipeline and cache strategy',
      signal: new AbortController().signal,
    };

    const decision = await strategy.route(
      mockContext,
      mockConfig,
      mockClient,
      mockLocalLiteRtLmClient,
    );

    expect(decision).not.toBeNull();
    expect(decision?.model).toBe(DEEPSEEK_REASONER_MODEL);
    expect(decision?.metadata.reasoning).toContain('85/70');
  });

  it('should route to deepseek-v4-flash when score < 70', async () => {
    const mockConfig = {
      getModel: () => DEEPSEEK_CHAT_MODEL,
    } as Config;

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [
          {
            message: {
              content: JSON.stringify({
                score: 30,
                reason: 'Single file helper function update',
              }),
            },
          },
        ],
      }),
    } as Response);

    const mockContext: RoutingContext = {
      history: [],
      request: 'Fix typo in index.ts',
      signal: new AbortController().signal,
    };

    const decision = await strategy.route(
      mockContext,
      mockConfig,
      mockClient,
      mockLocalLiteRtLmClient,
    );

    expect(decision).not.toBeNull();
    expect(decision?.model).toBe(DEEPSEEK_CHAT_MODEL);
    expect(decision?.metadata.reasoning).toContain('30/70');
  });

  it('should handle fetch errors gracefully and return null', async () => {
    const mockConfig = {
      getModel: () => DEEPSEEK_CHAT_MODEL,
    } as Config;

    globalThis.fetch = vi.fn().mockRejectedValue(new Error('Network error'));

    const mockContext: RoutingContext = {
      history: [],
      request: 'Implement a new caching system with LRU eviction',
      signal: new AbortController().signal,
    };

    const decision = await strategy.route(
      mockContext,
      mockConfig,
      mockClient,
      mockLocalLiteRtLmClient,
    );

    expect(decision).toBeNull();
  });
});
