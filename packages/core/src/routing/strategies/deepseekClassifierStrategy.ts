/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { BaseLlmClient } from '../../core/baseLlmClient.js';
import type {
  RoutingContext,
  RoutingDecision,
  RoutingStrategy,
} from '../routingStrategy.js';
import type { Config } from '../../config/config.js';
import { DEEPSEEK_CHAT_MODEL, DEEPSEEK_REASONER_MODEL } from '../../config/models.js';
import { debugLogger } from '../../utils/debugLogger.js';
import type { LocalLiteRtLmClient } from '../../core/localLiteRtLmClient.js';

// Threshold tuned to escalate to deepseek-v4-pro only for genuinely complex
// tasks (architecture, broad refactors, multi-file debugging). v4-pro costs
// ~5x more input and ~3x more output than v4-flash, plus thinking-mode
// reasoning tokens. A score of 70+ corresponds to "Complex" and "Strategic"
// in the classifier prompt — routine code changes should run on flash.
const COMPLEXITY_THRESHOLD = 70;

const CLASSIFIER_PROMPT = `You are a task complexity classifier. Analyze the user's request and assign a complexity score from 1 to 100.

Score guide:
1-20:  Trivial — single read/list operation, simple factual question
21-50: Routine — single file edit, simple fix, straightforward code generation
51-80: Complex — multi-file changes, debugging unknown causes, feature requiring broad context
81-100: Strategic — architecture design, large-scale refactoring, novel system design

Respond ONLY with valid JSON in this exact format:
{"score": <number 1-100>, "reason": "<one sentence>"}`;

export class DeepSeekClassifierStrategy implements RoutingStrategy {
  readonly name = 'deepseek_classifier';

  async route(
    context: RoutingContext,
    config: Config,
    _baseLlmClient: BaseLlmClient,
    _localLiteRtLmClient: LocalLiteRtLmClient,
  ): Promise<RoutingDecision | null> {
    const model = config.getModel();
    if (!model.startsWith('deepseek-')) {
      return null;
    }

    const apiKey = process.env['DEEPSEEK_API_KEY'];
    if (!apiKey) {
      return null;
    }

    const startTime = Date.now();
    try {
      const userText =
        typeof context.request === 'string'
          ? context.request
          : Array.isArray(context.request)
            ? context.request
                .map((p: any) => (typeof p === 'string' ? p : (p.text ?? '')))
                .join(' ')
            : String(context.request);

      const response = await fetch('https://api.deepseek.com/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: DEEPSEEK_CHAT_MODEL,
          messages: [
            { role: 'system', content: CLASSIFIER_PROMPT },
            { role: 'user', content: userText },
          ],
          response_format: { type: 'json_object' },
          max_tokens: 100,
          temperature: 0,
          stream: false,
          // The classifier is a lightweight "score 1-100" task — no
          // chain-of-thought needed. Disabling thinking mode skips
          // `reasoning_content` generation entirely, halving the latency
          // and cost of every routing decision.
          thinking: { type: 'disabled' },
        }),
        signal: context.signal,
      });

      if (!response.ok) {
        const errBody = await response.text();
        debugLogger.warn(`[DeepSeekRouter] Classification call failed: ${response.status} ${errBody}`);
        console.error(`[DeepSeekRouter] Classification HTTP error: ${response.status}`);
        return null;
      }

      const data: any = await response.json();
      const rawText: string = data.choices?.[0]?.message?.content ?? '';
      debugLogger.log(`[DeepSeekRouter] Raw classifier response: ${rawText}`);

      // Strip markdown code fences if present
      const jsonText = rawText.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim();
      let parsed: any;
      try {
        parsed = JSON.parse(jsonText);
      } catch (e) {
        console.error(`[DeepSeekRouter] Failed to parse classifier JSON: "${rawText}"`);
        return null;
      }
      const score: number = Number(parsed.score ?? 0);

      const selectedModel =
        score >= COMPLEXITY_THRESHOLD ? DEEPSEEK_REASONER_MODEL : DEEPSEEK_CHAT_MODEL;

      const latencyMs = Date.now() - startTime;
      debugLogger.log(
        `[DeepSeekRouter] score=${score} → ${selectedModel} (${latencyMs}ms): ${parsed.reason ?? ''}`,
      );

      return {
        model: selectedModel,
        metadata: {
          source: 'DeepSeekClassifier',
          latencyMs,
          reasoning: `[Score: ${score}/${COMPLEXITY_THRESHOLD}] ${parsed.reason ?? ''}`,
        },
      };
    } catch (error) {
      debugLogger.warn(`[DeepSeekRouter] Classification failed, using default:`, error);
      console.error(`[DeepSeekRouter] Caught error:`, error);
      return null;
    }
  }
}
