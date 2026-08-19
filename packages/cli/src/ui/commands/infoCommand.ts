/**
 * @license
 * Copyright 2026 sluisr
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  type CommandContext,
  CommandKind,
  type SlashCommand,
} from './types.js';

export const infoCommand: SlashCommand = {
  name: 'info',
  altNames: ['author', 'credits', 'links'],
  description: 'Show info about the DeepSeek CLI port, author, and social links.',
  kind: CommandKind.BUILT_IN,
  autoExecute: false,
  action: async (context: CommandContext, _args: string) => {
    const infoText = `
╭──────────────────────────────────────────────────────────────╮
│                 DeepSeek CLI (Unofficial)                    │
│             High-Performance Agentic AI Terminal             │
╰──────────────────────────────────────────────────────────────╯

  Developer:  sluisr
  Version:    DeepSeek V4 Edition
  License:    Copyright 2026 sluisr

  Links & Community:
  • Website:     https://sluisr.com/
  • GitHub:      https://github.com/sluisr
  • YouTube:     https://www.youtube.com/@sluisr_

  Port Highlights:
  • Native DeepSeek V4-Flash & V4-Pro (Reasoning / Thinking CoT)
  • Live dynamic reasoning effort & temperature hot-reloading
  • Fill-in-the-Middle code completion (\`/fim\`)
  • Chat Prefix Completion (\`/prefix\`)
  • Live DeepSeek account balance lookup (\`/balance\`)
  • Native server-side web search with citations
  • Unified diff code patching (\`apply_patch\`)
  • Silent 0ms-lag sudo/ssh password prompts (\`askpass\`)
`.trim();

    context.ui.addItem({
      type: 'info',
      text: infoText,
      icon: '',
    } as any);
  },
};
