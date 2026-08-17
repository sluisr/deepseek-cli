/**
 * @license
 * Copyright 2026 Google LLC
 * Copyright 2026 sluisr
 * SPDX-License-Identifier: Apache-2.0
 */

import * as fsPromises from 'node:fs/promises';
import * as path from 'node:path';
import {
  type CommandContext,
  CommandKind,
  type SlashCommand,
} from './types.js';
import { MessageType } from '../types.js';
import {
  resolveDefensiveToolPath,
  loadDeepSeekApiKey,
} from '@google/gemini-cli-core';

export const fimCommand: SlashCommand = {
  name: 'fim',
  altNames: ['fill'],
  description:
    'DeepSeek Fill-in-the-Middle code completion. Usage: /fim <file> [<FIM_HOLE> | <line>]',
  kind: CommandKind.BUILT_IN,
  autoExecute: false,
  action: async (context: CommandContext, args: string) => {
    const rawArgs = args.trim().split(/\s+/).filter(Boolean);
    if (rawArgs.length === 0) {
      context.ui.addItem({
        type: MessageType.INFO,
        text: `DeepSeek Fill-in-the-Middle (FIM) Completion:
Usage: /fim <filepath> [line_number]

Place <FIM_HOLE> or // FIM inside your code where you want DeepSeek to fill in the implementation, then run:
  /fim path/to/file.ts

Or specify a line number to insert code at:
  /fim path/to/file.ts 42`,
      });
      return;
    }

    const filePath = rawArgs[0];
    const lineArg = rawArgs.length > 1 ? parseInt(rawArgs[1], 10) : NaN;

    const config = context.services.agentContext?.config;
    const targetDir = config?.getTargetDir() || process.cwd();
    const resolvedPath = resolveDefensiveToolPath(targetDir, filePath);

    let content: string;
    try {
      content = await fsPromises.readFile(resolvedPath, 'utf-8');
    } catch (err: any) {
      context.ui.addItem({
        type: MessageType.ERROR,
        text: `Could not read file "${filePath}": ${err?.message || err}`,
      });
      return;
    }

    let prefix = '';
    let suffix = '';
    let holeDescription = '';

    const MARKERS = ['<FIM_HOLE>', '<FILL_HERE>', '// FIM', '/* FIM */', '# FIM'];
    let markerFound = '';

    for (const m of MARKERS) {
      if (content.includes(m)) {
        markerFound = m;
        break;
      }
    }

    if (markerFound) {
      const parts = content.split(markerFound);
      prefix = parts[0];
      suffix = parts.slice(1).join(markerFound);
      holeDescription = `at marker "${markerFound}"`;
    } else if (!isNaN(lineArg) && lineArg > 0) {
      const lines = content.split('\n');
      prefix = lines.slice(0, lineArg - 1).join('\n') + '\n';
      suffix = '\n' + lines.slice(lineArg - 1).join('\n');
      holeDescription = `at line ${lineArg}`;
    } else {
      context.ui.addItem({
        type: MessageType.ERROR,
        text: `No <FIM_HOLE> marker found in "${filePath}". Please add <FIM_HOLE> or specify a line number (e.g. /fim ${filePath} 25).`,
      });
      return;
    }

    const apiKey =
      config?.getContentGeneratorConfig()?.apiKey ||
      process.env['DEEPSEEK_API_KEY'] ||
      (await loadDeepSeekApiKey());
    if (!apiKey) {
      context.ui.addItem({
        type: MessageType.ERROR,
        text: 'Missing DeepSeek API key for FIM completion. Set DEEPSEEK_API_KEY or authenticate.',
      });
      return;
    }

    context.ui.addItem({
      type: MessageType.INFO,
      text: `Generating FIM code completion ${holeDescription} in ${path.basename(resolvedPath)}...`,
    });

    try {
      const res = await fetch('https://api.deepseek.com/beta/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: 'deepseek-chat',
          prompt: prefix,
          suffix: suffix,
          temperature: config?.getTemperature?.() ?? 0.0,
          max_tokens: 4096,
        }),
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`DeepSeek API error (${res.status}): ${errorText}`);
      }

      const data: any = await res.json();
      const completionText = data.choices?.[0]?.text ?? '';

      if (!completionText) {
        context.ui.addItem({
          type: MessageType.INFO,
          text: 'DeepSeek returned an empty completion for this context.',
        });
        return;
      }

      // Reconstruct the file with completion
      const newFileContent = prefix + completionText + suffix;
      await fsPromises.writeFile(resolvedPath, newFileContent, 'utf-8');

      context.ui.addItem({
        type: MessageType.INFO,
        text: `Successfully completed code in ${filePath}!\n\nInserted:\n\`\`\`\n${completionText}\n\`\`\``,
      });
    } catch (err: any) {
      context.ui.addItem({
        type: MessageType.ERROR,
        text: `FIM completion failed: ${err?.message || err}`,
      });
    }
  },
};
