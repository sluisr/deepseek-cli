import {
  ModelSlashCommandEvent,
  logModelSlashCommand,
  DEEPSEEK_CHAT_MODEL,
  DEEPSEEK_REASONER_MODEL,
} from '@google/gemini-cli-core';
import {
  type CommandContext,
  CommandKind,
  type SlashCommand,
} from './types.js';
import { MessageType } from '../types.js';

function resolveModelShortcut(input: string): string {
  const lower = input.toLowerCase().trim();
  if (lower === 'pro' || lower === 'v4-pro' || lower === 'deepseek-pro') {
    return DEEPSEEK_REASONER_MODEL;
  }
  if (lower === 'flash' || lower === 'v4-flash' || lower === 'deepseek-flash') {
    return DEEPSEEK_CHAT_MODEL;
  }
  return input.trim();
}

const setModelCommand: SlashCommand = {
  name: 'set',
  description:
    'Set the model to use. Usage: /model set <model-name> [--persist]',
  kind: CommandKind.BUILT_IN,
  autoExecute: false,
  action: async (context: CommandContext, args: string) => {
    const parts = args.trim().split(/\s+/).filter(Boolean);
    if (parts.length === 0) {
      context.ui.addItem({
        type: MessageType.ERROR,
        text: 'Usage: /model set <model-name> [--persist]',
      });
      return;
    }

    const modelName = resolveModelShortcut(parts[0]);
    const persist = parts.includes('--persist');

    if (context.services.agentContext?.config) {
      context.services.agentContext.config.setModel(modelName, !persist);
      const event = new ModelSlashCommandEvent(modelName);
      logModelSlashCommand(context.services.agentContext.config, event);

      context.ui.addItem({
        type: MessageType.INFO,
        text: `Model set to ${modelName}${persist ? ' (persisted)' : ''}`,
      });
    }
  },
};

const manageModelCommand: SlashCommand = {
  name: 'manage',
  description: 'Opens a dialog to configure the model',
  kind: CommandKind.BUILT_IN,
  autoExecute: true,
  action: async (context: CommandContext) => {
    if (context.services.agentContext?.config) {
      await context.services.agentContext.config.refreshUserQuota();
    }
    return {
      type: 'dialog',
      dialog: 'model',
    };
  },
};

export const modelCommand: SlashCommand = {
  name: 'model',
  description: 'Manage model configuration',
  kind: CommandKind.BUILT_IN,
  autoExecute: false,
  subCommands: [manageModelCommand, setModelCommand],
  action: async (context: CommandContext, args: string) => {
    const trimmed = args.trim();
    if (!trimmed) {
      return manageModelCommand.action!(context, args);
    }
    // Direct shortcut support: `/model pro`, `/model flash`, `/model deepseek-v4-pro`
    if (trimmed.startsWith('set ')) {
      return setModelCommand.action!(context, trimmed.slice(4));
    }
    return setModelCommand.action!(context, trimmed);
  },
};

