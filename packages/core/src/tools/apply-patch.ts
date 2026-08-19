/**
 * @license
 * Copyright 2026 Google LLC
 * Copyright 2026 sluisr
 * SPDX-License-Identifier: Apache-2.0
 */

import * as fsPromises from 'node:fs/promises';
import * as path from 'node:path';
import * as Diff from 'diff';
import {
  BaseDeclarativeTool,
  BaseToolInvocation,
  Kind,
  type ToolInvocation,
  type ToolResult,
  type ExecuteOptions,
} from './tools.js';
import { ToolErrorType } from './tool-error.js';
import { resolveDefensiveToolPath } from '../utils/paths.js';
import { isNodeError, getErrorMessage } from '../utils/errors.js';
import type { AgentLoopContext } from '../config/agent-loop-context.js';
import type { MessageBus } from '../confirmation-bus/message-bus.js';

export const APPLY_PATCH_TOOL_NAME = 'apply_patch';
export const APPLY_PATCH_DISPLAY_NAME = 'Apply Patch';

export interface ApplyPatchParams {
  /**
   * The patch content in standard unified diff format.
   * Can contain multiple files.
   */
  patch: string;
}

export interface ApplyPatchResult extends ToolResult {
  appliedFiles?: string[];
  failedFiles?: string[];
}

class ApplyPatchInvocation extends BaseToolInvocation<
  ApplyPatchParams,
  ApplyPatchResult
> {
  constructor(
    private readonly context: AgentLoopContext,
    params: ApplyPatchParams,
    messageBus: MessageBus,
  ) {
    super(params, messageBus, APPLY_PATCH_TOOL_NAME, APPLY_PATCH_DISPLAY_NAME);
  }

  getDescription(): string {
    return 'apply patch';
  }

  async execute(_options?: ExecuteOptions): Promise<ApplyPatchResult> {
    const rawPatch = this.params.patch;
    if (!rawPatch || !rawPatch.trim()) {
      return {
        llmContent: 'Error: Patch content is empty.',
        returnDisplay: 'Empty patch provided.',
        error: {
          message: 'Patch content cannot be empty.',
          type: ToolErrorType.INVALID_TOOL_PARAMS,
        },
      };
    }

    const appliedFiles: string[] = [];
    const failedFiles: string[] = [];
    const logs: string[] = [];

    try {
      const parsedPatches = Diff.parsePatch(rawPatch);
      if (parsedPatches.length === 0) {
        // Fallback: try parsing as a single file raw diff if header was missing
        return {
          llmContent: `Error: Unable to parse patch. Ensure it follows unified diff format (*** filename / --- filename / @@ ... @@).`,
          returnDisplay: 'Failed to parse patch.',
          error: {
            message: 'Invalid patch format.',
            type: ToolErrorType.INVALID_TOOL_PARAMS,
          },
        };
      }

      for (const p of parsedPatches) {
        const rawFileName = p.newFileName || p.oldFileName || '';
        const cleanedFileName = rawFileName.replace(/^[ab]\//, '').replace(/^\/dev\/null/, '').trim();

        if (!cleanedFileName) {
          failedFiles.push('unknown file');
          logs.push('Could not determine target filename from patch header.');
          continue;
        }

        const targetPath = resolveDefensiveToolPath(
          this.context.config.getTargetDir(),
          cleanedFileName,
        );

        let fileContent = '';
        let fileExists = false;
        try {
          fileContent = await fsPromises.readFile(targetPath, 'utf-8');
          fileExists = true;
        } catch (err) {
          if (!isNodeError(err) || err.code !== 'ENOENT') {
            throw err;
          }
        }

        // Apply patch hunks using jsdiff
        const patched = Diff.applyPatch(fileContent, p, {
          fuzzFactor: 2,
        });

        if (patched === false) {
          failedFiles.push(cleanedFileName);
          logs.push(`Failed to apply patch hunks to ${cleanedFileName}. Context mismatch.`);
          continue;
        }

        // Write patched content
        await fsPromises.mkdir(path.dirname(targetPath), { recursive: true });
        await fsPromises.writeFile(targetPath, patched, 'utf-8');

        appliedFiles.push(cleanedFileName);
        const action = fileExists ? 'Patched' : 'Created';
        logs.push(`${action} ${cleanedFileName} successfully.`);
      }

      const success = appliedFiles.length > 0 && failedFiles.length === 0;
      return {
        llmContent: logs.join('\n'),
        returnDisplay: success
          ? `Successfully applied patch to ${appliedFiles.join(', ')}.`
          : `Applied patch with errors: ${logs.join('; ')}`,
        appliedFiles,
        failedFiles,
      };
    } catch (err) {
      const msg = getErrorMessage(err);
      return {
        llmContent: `Error applying patch: ${msg}`,
        returnDisplay: 'Patch application failed.',
        error: {
          message: msg,
          type: ToolErrorType.EXECUTION_FAILED,
        },
      };
    }
  }
}

export class ApplyPatchTool extends BaseDeclarativeTool<
  ApplyPatchParams,
  ApplyPatchResult
> {
  static readonly Name = APPLY_PATCH_TOOL_NAME;

  constructor(
    private readonly context: AgentLoopContext,
    messageBus: MessageBus,
  ) {
    super(
      ApplyPatchTool.Name,
      APPLY_PATCH_DISPLAY_NAME,
      'Applies unified diff patches directly to files. Fast and token-efficient for code editing.',
      Kind.Edit,
      {
        type: 'object',
        properties: {
          patch: {
            type: 'string',
            description: 'The unified diff patch content to apply.',
          },
        },
        required: ['patch'],
      },
      messageBus,
    );
  }

  protected createInvocation(
    params: ApplyPatchParams,
    messageBus: MessageBus,
  ): ToolInvocation<ApplyPatchParams, ApplyPatchResult> {
    return new ApplyPatchInvocation(this.context, params, messageBus);
  }
}
