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

export const prefixCommand: SlashCommand = {
  name: 'prefix',
  altNames: ['prefill', 'continue-from'],
  description:
    'DeepSeek Chat Prefix Completion. Pre-fill the assistant response for the next prompt.',
  kind: CommandKind.BUILT_IN,
  autoExecute: false,
  action: async (context: CommandContext, args: string) => {
    const config = context.services.agentContext?.config as any;
    const trimmed = args.trim();

    if (trimmed === 'clear' || trimmed === 'reset') {
      config?.clearAssistantPrefix?.();
      context.ui.addItem({
        type: MessageType.INFO,
        text: 'Assistant prefix cleared.',
      });
      return;
    }

    if (!trimmed) {
      const activePrefix = config?.getAssistantPrefix?.();
      if (activePrefix) {
        context.ui.addItem({
          type: MessageType.INFO,
          text: `Active assistant prefix: "${activePrefix}"\nRun \`/prefix clear\` to cancel.`,
        });
      } else {
        context.ui.addItem({
          type: MessageType.INFO,
          text: `DeepSeek Chat Prefix Completion (Beta):
Pre-fill the assistant's starting text for the next prompt. The model will continue directly from this exact prefix without conversational intros or wrappers.

Usage:
  /prefix <prefix_text>

Examples:
  /prefix \`\`\`json
  /prefix {"status": "success", "result":
  /prefix \`\`\`typescript\\nexport async function

To clear:
  /prefix clear`,
        });
      }
      return;
    }

    // Set the prefix
    config?.setAssistantPrefix?.(trimmed);

    context.ui.addItem({
      type: MessageType.INFO,
      text: `✅ Assistant prefix set for your next prompt:\n\`\`\`\n${trimmed}\n\`\`\`\nNow send your message and DeepSeek will continue directly from this prefix!`,
    });
  },
};
