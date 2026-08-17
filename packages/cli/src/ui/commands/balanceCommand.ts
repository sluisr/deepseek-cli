/**
 * @license
 * Copyright 2026 Google LLC
 * Copyright 2026 sluisr
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  type CommandContext,
  CommandKind,
  type SlashCommand,
} from './types.js';
import { MessageType } from '../types.js';
import { loadDeepSeekApiKey } from '@google/gemini-cli-core';

export const balanceCommand: SlashCommand = {
  name: 'balance',
  altNames: ['wallet', 'credits'],
  description:
    'Check your DeepSeek account balance, topped-up funds, and granted credits.',
  kind: CommandKind.BUILT_IN,
  autoExecute: false,
  action: async (context: CommandContext, _args: string) => {
    const config = context.services.agentContext?.config;
    const apiKey =
      config?.getContentGeneratorConfig()?.apiKey ||
      process.env['DEEPSEEK_API_KEY'] ||
      (await loadDeepSeekApiKey());

    if (!apiKey) {
      context.ui.addItem({
        type: MessageType.ERROR,
        text: 'Missing DeepSeek API key. Please authenticate or set DEEPSEEK_API_KEY.',
      });
      return;
    }

    try {
      const res = await fetch('https://api.deepseek.com/user/balance', {
        method: 'GET',
        headers: {
          Accept: 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },
      });

      if (!res.ok) {
        const errText = await res.text();
        context.ui.addItem({
          type: MessageType.ERROR,
          text: `Failed to fetch balance (${res.status}): ${errText}`,
        });
        return;
      }

      const data: any = await res.json();
      const isAvailable = data.is_available ?? true;
      const statusIcon = isAvailable ? 'Active' : 'Exhausted / Unavailable';

      const balanceInfos: Array<{
        currency?: string;
        total_balance?: string;
        granted_balance?: string;
        topped_up_balance?: string;
      }> = data.balance_infos || [];

      let detailsText = `DeepSeek Account Balance\n`;
      detailsText += `Account Status: ${statusIcon}\n`;

      if (balanceInfos.length === 0) {
        detailsText += `\nNo currency balance details returned.`;
      } else {
        for (const info of balanceInfos) {
          const curr = info.currency || 'USD';
          const sym = curr === 'USD' ? '$' : '¥';
          const totalVal = info.total_balance ?? '0.00';
          const grantedVal = parseFloat(info.granted_balance ?? '0');
          const toppedUpVal = info.topped_up_balance ?? '0.00';

          detailsText += `\n───────────── [ ${curr} ] ─────────────\n`;
          if (grantedVal > 0) {
            detailsText += `  • Total Balance:     ${sym}${totalVal}\n`;
            detailsText += `  • Topped-Up Funds:   ${sym}${toppedUpVal}\n`;
            detailsText += `  • Granted Credits:   ${sym}${info.granted_balance}\n`;
          } else {
            detailsText += `  • Available Balance: ${sym}${totalVal}\n`;
          }
        }
      }

      context.ui.addItem({
        type: MessageType.INFO,
        text: detailsText.trimEnd(),
      });
    } catch (err: any) {
      context.ui.addItem({
        type: MessageType.ERROR,
        text: `Error checking DeepSeek balance: ${err?.message || err}`,
      });
    }
  },
};
