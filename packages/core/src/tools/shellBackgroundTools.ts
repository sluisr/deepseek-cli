/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import fs from 'node:fs';
import { ShellExecutionService } from '../services/shellExecutionService.js';
import { ExecutionLifecycleService } from '../services/executionLifecycleService.js';
import {
  BaseDeclarativeTool,
  BaseToolInvocation,
  Kind,
  type ToolResult,
  type ExecuteOptions,
} from './tools.js';

import { ToolErrorType } from './tool-error.js';
import type { MessageBus } from '../confirmation-bus/message-bus.js';
import type { AgentLoopContext } from '../config/agent-loop-context.js';
import { isNodeError } from '../utils/errors.js';

const MAX_BUFFER_LOAD_CAP_BYTES = 64 * 1024; // Safe 64KB buffer load Cap
const DEFAULT_TAIL_LINES_COUNT = 100;

// --- list_background_processes ---

class ListBackgroundProcessesInvocation extends BaseToolInvocation<
  Record<string, never>,
  ToolResult
> {
  constructor(
    private readonly context: AgentLoopContext,
    params: Record<string, never>,
    messageBus: MessageBus,
    toolName?: string,
    toolDisplayName?: string,
  ) {
    super(params, messageBus, toolName, toolDisplayName);
  }

  getDescription(): string {
    return 'Lists all active and recently completed background processes for the current session.';
  }

  async execute({ abortSignal: _signal }: ExecuteOptions): Promise<ToolResult> {
    const processes = ShellExecutionService.listBackgroundProcesses(
      this.context.config.getSessionId(),
    );
    if (processes.length === 0) {
      return {
        llmContent: 'No background processes found.',
        returnDisplay: 'No background processes found.',
      };
    }

    const lines = processes.map(
      (p) =>
        `- [PID ${p.pid}] ${p.status.toUpperCase()}: \`${p.command}\`${
          p.exitCode !== undefined ? ` (Exit Code: ${p.exitCode})` : ''
        }${p.signal ? ` (Signal: ${p.signal})` : ''}`,
    );

    const content = lines.join('\n');
    return {
      llmContent: content,
      returnDisplay: content,
    };
  }
}

export class ListBackgroundProcessesTool extends BaseDeclarativeTool<
  Record<string, never>,
  ToolResult
> {
  static readonly Name = 'list_background_processes';

  constructor(
    private readonly context: AgentLoopContext,
    messageBus: MessageBus,
  ) {
    super(
      ListBackgroundProcessesTool.Name,
      'List Background Processes',
      'Lists all active and recently completed background shell processes orchestrating by the agent.',
      Kind.Read,
      {
        type: 'object',
        properties: {},
      },
      messageBus,
    );
  }

  protected createInvocation(
    params: Record<string, never>,
    messageBus: MessageBus,
  ) {
    return new ListBackgroundProcessesInvocation(
      this.context,
      params,
      messageBus,
      this.name,
    );
  }
}

// --- read_background_output ---

interface ReadBackgroundOutputParams {
  pid: number;
  lines?: number;
  delay_ms?: number;
}

class ReadBackgroundOutputInvocation extends BaseToolInvocation<
  ReadBackgroundOutputParams,
  ToolResult
> {
  constructor(
    private readonly context: AgentLoopContext,
    params: ReadBackgroundOutputParams,
    messageBus: MessageBus,
    toolName?: string,
    toolDisplayName?: string,
  ) {
    super(params, messageBus, toolName, toolDisplayName);
  }

  getDescription(): string {
    return `Reading output for background process ${this.params.pid}`;
  }

  async execute({ abortSignal: _signal }: ExecuteOptions): Promise<ToolResult> {
    const pid = this.params.pid;

    if (this.params.delay_ms && this.params.delay_ms > 0) {
      await new Promise((resolve) => setTimeout(resolve, this.params.delay_ms));
    }

    // Verify process belongs to this session to prevent reading logs of processes from other sessions/users
    const processes = ShellExecutionService.listBackgroundProcesses(
      this.context.config.getSessionId(),
    );
    if (!processes.some((p) => p.pid === pid)) {
      return {
        llmContent: `Access denied. Background process ID ${pid} not found in this session's history.`,
        returnDisplay: 'Access denied.',
        error: {
          message: `Background process history lookup failed for PID ${pid}`,
          type: ToolErrorType.EXECUTION_FAILED,
        },
      };
    }

    const logPath = ShellExecutionService.getLogFilePath(pid);

    try {
      await fs.promises.access(logPath);
    } catch {
      return {
        llmContent: `No output log found for process ID ${pid}. It might not have produced output or was cleaned up.`,
        returnDisplay: `No log found for PID ${pid}`,
        error: {
          message: `Log file not found at ${logPath}`,
          type: ToolErrorType.EXECUTION_FAILED,
        },
      };
    }

    try {
      const fileHandle = await fs.promises.open(
        logPath,
        fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW,
      );

      let content = '';
      let position = 0;
      try {
        const stats = await fileHandle.stat();
        const readSize = Math.min(stats.size, MAX_BUFFER_LOAD_CAP_BYTES);
        position = Math.max(0, stats.size - readSize);

        const buffer = Buffer.alloc(readSize);
        await fileHandle.read(buffer, 0, readSize, position);
        content = buffer.toString('utf-8');
      } finally {
        await fileHandle.close();
      }

      if (!content) {
        return {
          llmContent: 'Log is empty.',
          returnDisplay: 'Log is empty.',
        };
      }

      const logLines = content.split('\n');
      if (logLines.length > 0 && logLines[logLines.length - 1] === '') {
        logLines.pop();
      }

      // Discard first line if we started reading from middle of file to avoid partial lines
      if (position > 0 && logLines.length > 0) {
        logLines.shift();
      }

      const requestedLinesCount = this.params.lines ?? DEFAULT_TAIL_LINES_COUNT;
      const tailLines = logLines.slice(-requestedLinesCount);
      const output = tailLines.join('\n');

      const header =
        requestedLinesCount < logLines.length
          ? `Showing last ${requestedLinesCount} of ${logLines.length} lines:\n`
          : 'Full Log Output:\n';

      const responseContent = header + output;

      return {
        llmContent: responseContent,
        returnDisplay: responseContent,
      };
    } catch (error) {
      if (isNodeError(error) && error.code === 'ELOOP') {
        return {
          llmContent:
            'Symbolic link detected at predicted log path. Access is denied for security reasons.',
          returnDisplay: `Symlink detected for PID ${pid}`,
          error: {
            message:
              'Symbolic link detected at predicted log path. Access is denied for security reasons.',
            type: ToolErrorType.EXECUTION_FAILED,
          },
        };
      }
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return {
        llmContent: `Error reading background log: ${errorMessage}`,
        returnDisplay: 'Failed to read log.',
        error: {
          message: errorMessage,
          type: ToolErrorType.EXECUTION_FAILED,
        },
      };
    }
  }
}

export class ReadBackgroundOutputTool extends BaseDeclarativeTool<
  ReadBackgroundOutputParams,
  ToolResult
> {
  static readonly Name = 'read_background_output';

  constructor(
    private readonly context: AgentLoopContext,
    messageBus: MessageBus,
  ) {
    super(
      ReadBackgroundOutputTool.Name,
      'Read Background Output',
      'Reads the output log of a background shell process. Support reading tail snapshot.',
      Kind.Read,
      {
        type: 'object',
        properties: {
          pid: {
            type: 'integer',
            description:
              'The process ID (PID) of the background process to inspect.',
          },
          lines: {
            type: 'integer',
            minimum: 1,
            description:
              'Optional. Number of lines to read from the end of the log. Defaults to 100.',
          },
          delay_ms: {
            type: 'integer',
            description:
              'Optional. Delay in milliseconds to wait before reading the output. Useful to allow the process to start and generate initial output.',
          },
        },
        required: ['pid'],
      },
      messageBus,
    );
  }

  protected createInvocation(
    params: ReadBackgroundOutputParams,
    messageBus: MessageBus,
  ) {
    return new ReadBackgroundOutputInvocation(
      this.context,
      params,
      messageBus,
      this.name,
    );
  }
}

// --- kill_background_process ---

interface KillBackgroundProcessParams {
  pid: number;
}

class KillBackgroundProcessInvocation extends BaseToolInvocation<
  KillBackgroundProcessParams,
  ToolResult
> {
  constructor(
    private readonly context: AgentLoopContext,
    params: KillBackgroundProcessParams,
    messageBus: MessageBus,
    toolName?: string,
    toolDisplayName?: string,
  ) {
    super(params, messageBus, toolName, toolDisplayName);
  }

  getDescription(): string {
    return `Terminating background process with PID ${this.params.pid}`;
  }

  async execute({ abortSignal: _signal }: ExecuteOptions): Promise<ToolResult> {
    const pid = this.params.pid;
    const sessionId = this.context.config.getSessionId();
    const processes = ShellExecutionService.listBackgroundProcesses(sessionId);
    const targetProcess = processes.find((p) => p.pid === pid);

    if (!targetProcess) {
      return {
        llmContent: `Background process with PID ${pid} was not found in this session.`,
        returnDisplay: `PID ${pid} not found.`,
        error: {
          message: `Process ${pid} not found`,
          type: ToolErrorType.EXECUTION_FAILED,
        },
      };
    }

    if (
      targetProcess.status === 'exited' &&
      !ExecutionLifecycleService.isActive(pid)
    ) {
      return {
        llmContent: `Background process with PID ${pid} has already exited.`,
        returnDisplay: `PID ${pid} already exited.`,
      };
    }

    try {
      ExecutionLifecycleService.kill(pid);
      return {
        llmContent: `Successfully terminated background process with PID ${pid} (\`${targetProcess.command}\`).`,
        returnDisplay: `Killed process ${pid}.`,
      };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return {
        llmContent: `Failed to terminate background process ${pid}: ${errorMessage}`,
        returnDisplay: `Failed to kill PID ${pid}.`,
        error: {
          message: errorMessage,
          type: ToolErrorType.EXECUTION_FAILED,
        },
      };
    }
  }
}

export class KillBackgroundProcessTool extends BaseDeclarativeTool<
  KillBackgroundProcessParams,
  ToolResult
> {
  static readonly Name = 'kill_background_process';

  constructor(
    private readonly context: AgentLoopContext,
    messageBus: MessageBus,
  ) {
    super(
      KillBackgroundProcessTool.Name,
      'Kill Background Process',
      'Terminates a background shell process or hung task by its process ID (PID).',
      Kind.Execute,
      {
        type: 'object',
        properties: {
          pid: {
            type: 'integer',
            description:
              'The process ID (PID) of the background process to terminate.',
          },
        },
        required: ['pid'],
      },
      messageBus,
    );
  }

  protected createInvocation(
    params: KillBackgroundProcessParams,
    messageBus: MessageBus,
  ) {
    return new KillBackgroundProcessInvocation(
      this.context,
      params,
      messageBus,
      this.name,
    );
  }
}

// --- write_background_input ---

interface WriteBackgroundInputParams {
  pid: number;
  input: string;
}

class WriteBackgroundInputInvocation extends BaseToolInvocation<
  WriteBackgroundInputParams,
  ToolResult
> {
  constructor(
    private readonly context: AgentLoopContext,
    params: WriteBackgroundInputParams,
    messageBus: MessageBus,
    toolName?: string,
    toolDisplayName?: string,
  ) {
    super(params, messageBus, toolName, toolDisplayName);
  }

  getDescription(): string {
    return `Sending input to background process with PID ${this.params.pid}`;
  }

  async execute({ abortSignal: _signal }: ExecuteOptions): Promise<ToolResult> {
    const pid = this.params.pid;
    const sessionId = this.context.config.getSessionId();
    const processes = ShellExecutionService.listBackgroundProcesses(sessionId);
    const targetProcess = processes.find((p) => p.pid === pid);

    if (!targetProcess) {
      return {
        llmContent: `Background process with PID ${pid} was not found in this session.`,
        returnDisplay: `PID ${pid} not found.`,
        error: {
          message: `Process ${pid} not found`,
          type: ToolErrorType.EXECUTION_FAILED,
        },
      };
    }

    if (!ExecutionLifecycleService.isActive(pid)) {
      return {
        llmContent: `Background process with PID ${pid} is not active.`,
        returnDisplay: `PID ${pid} is not active.`,
        error: {
          message: `Process ${pid} is not active`,
          type: ToolErrorType.EXECUTION_FAILED,
        },
      };
    }

    try {
      const formattedInput = this.params.input.endsWith('\n')
        ? this.params.input
        : `${this.params.input}\n`;

      ExecutionLifecycleService.writeInput(pid, formattedInput);
      return {
        llmContent: `Successfully sent input to background process ${pid}.`,
        returnDisplay: `Sent input to PID ${pid}.`,
      };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return {
        llmContent: `Failed to write input to background process ${pid}: ${errorMessage}`,
        returnDisplay: `Failed to write input to PID ${pid}.`,
        error: {
          message: errorMessage,
          type: ToolErrorType.EXECUTION_FAILED,
        },
      };
    }
  }
}

export class WriteBackgroundInputTool extends BaseDeclarativeTool<
  WriteBackgroundInputParams,
  ToolResult
> {
  static readonly Name = 'write_background_input';

  constructor(
    private readonly context: AgentLoopContext,
    messageBus: MessageBus,
  ) {
    super(
      WriteBackgroundInputTool.Name,
      'Write Background Input',
      'Sends stdin input to an active background process (e.g. answering interactive prompts or sending commands).',
      Kind.Execute,
      {
        type: 'object',
        properties: {
          pid: {
            type: 'integer',
            description:
              'The process ID (PID) of the active background process.',
          },
          input: {
            type: 'string',
            description: 'The text string to send to the process stdin.',
          },
        },
        required: ['pid', 'input'],
      },
      messageBus,
    );
  }

  protected createInvocation(
    params: WriteBackgroundInputParams,
    messageBus: MessageBus,
  ) {
    return new WriteBackgroundInputInvocation(
      this.context,
      params,
      messageBus,
      this.name,
    );
  }
}
