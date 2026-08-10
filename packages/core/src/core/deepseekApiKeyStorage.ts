/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { HybridTokenStorage } from '../mcp/token-storage/hybrid-token-storage.js';
import type { OAuthCredentials } from '../mcp/token-storage/types.js';
import { debugLogger } from '../utils/debugLogger.js';
import { createCache } from '../utils/cache.js';

const KEYCHAIN_SERVICE_NAME = 'gemini-cli-deepseek-key';
const DEFAULT_KEY_ENTRY = 'default-deepseek-key';

const storage = new HybridTokenStorage(KEYCHAIN_SERVICE_NAME);

const deepseekKeyCache = createCache<string, Promise<string | null>>({
  storage: 'map',
  defaultTtl: 30000,
});

export async function loadDeepSeekApiKey(): Promise<string | null> {
  return deepseekKeyCache.getOrCreate(DEFAULT_KEY_ENTRY, async () => {
    try {
      const credentials = await storage.getCredentials(DEFAULT_KEY_ENTRY);
      if (credentials?.token?.accessToken) {
        return credentials.token.accessToken;
      }
      return null;
    } catch (error: unknown) {
      debugLogger.error('Failed to load DeepSeek API key from storage:', error);
      return null;
    }
  });
}

export async function saveDeepSeekApiKey(
  apiKey: string | null | undefined,
): Promise<void> {
  deepseekKeyCache.delete(DEFAULT_KEY_ENTRY);
  if (!apiKey || apiKey.trim() === '') {
    try {
      await storage.deleteCredentials(DEFAULT_KEY_ENTRY);
    } catch (error: unknown) {
      debugLogger.warn('Failed to delete DeepSeek API key from storage:', error);
    }
    return;
  }

  const credentials: OAuthCredentials = {
    serverName: DEFAULT_KEY_ENTRY,
    token: {
      accessToken: apiKey,
      tokenType: 'ApiKey',
    },
    updatedAt: Date.now(),
  };

  await storage.setCredentials(credentials);
}

export async function clearDeepSeekApiKey(): Promise<void> {
  deepseekKeyCache.delete(DEFAULT_KEY_ENTRY);
  try {
    await storage.deleteCredentials(DEFAULT_KEY_ENTRY);
  } catch (error: unknown) {
    debugLogger.error('Failed to clear DeepSeek API key from storage:', error);
  }
}
