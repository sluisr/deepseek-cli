/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { ShellExecutionService } from '../services/shellExecutionService.js';
import { ExecutionLifecycleService } from '../services/executionLifecycleService.js';
import {
  KillBackgroundProcessTool,
  WriteBackgroundInputTool,
} from './shellBackgroundTools.js';
import { createMockMessageBus } from '../test-utils/mock-message-bus.js';
import type { AgentLoopContext } from '../config/agent-loop-context.js';

describe('Background Process Management Tools (Kill & Write Input)', () => {
  let killTool: KillBackgroundProcessTool;
  let writeTool: WriteBackgroundInputTool;
  const bus = createMockMessageBus();

  beforeEach(() => {
    vi.restoreAllMocks();
    const mockContext = {
      config: { getSessionId: () => 'default' },
    } as unknown as AgentLoopContext;
    killTool = new KillBackgroundProcessTool(mockContext, bus);
    writeTool = new WriteBackgroundInputTool(mockContext, bus);

    // Reset services
    (ShellExecutionService as any).backgroundProcessHistory.clear();
    ExecutionLifecycleService.resetForTest();
  });

  describe('kill_background_process', () => {
    it('should return error if process is not found in session history', async () => {
      const invocation = killTool.build({ pid: 12345 });
      (invocation as any).context = { config: { getSessionId: () => 'default' } };
      const result = await invocation.execute({
        abortSignal: new AbortController().signal,
      });

      expect(result.error).toBeDefined();
      expect(result.llmContent).toContain('not found in this session');
    });

    it('should inform if process is already exited', async () => {
      const pid = 54321;
      const history = new Map();
      history.set(pid, {
        command: 'sleep 10',
        status: 'exited',
        exitCode: 0,
        startTime: Date.now(),
      });
      (ShellExecutionService as any).backgroundProcessHistory.set(
        'default',
        history,
      );

      const invocation = killTool.build({ pid });
      (invocation as any).context = { config: { getSessionId: () => 'default' } };
      const result = await invocation.execute({
        abortSignal: new AbortController().signal,
      });

      expect(result.llmContent).toContain('has already exited');
    });

    it('should kill active background process successfully', async () => {
      const pid = 99999;
      const history = new Map();
      history.set(pid, {
        command: 'npm run dev',
        status: 'running',
        startTime: Date.now(),
      });
      (ShellExecutionService as any).backgroundProcessHistory.set(
        'default',
        history,
      );

      const killSpy = vi.spyOn(ExecutionLifecycleService, 'kill').mockImplementation(() => {});

      const invocation = killTool.build({ pid });
      (invocation as any).context = { config: { getSessionId: () => 'default' } };
      const result = await invocation.execute({
        abortSignal: new AbortController().signal,
      });

      expect(killSpy).toHaveBeenCalledWith(pid);
      expect(result.llmContent).toContain('Successfully terminated background process with PID 99999');
    });
  });

  describe('write_background_input', () => {
    it('should return error if process not found', async () => {
      const invocation = writeTool.build({ pid: 11111, input: 'yes' });
      (invocation as any).context = { config: { getSessionId: () => 'default' } };
      const result = await invocation.execute({
        abortSignal: new AbortController().signal,
      });

      expect(result.error).toBeDefined();
      expect(result.llmContent).toContain('not found in this session');
    });

    it('should return error if process is not active', async () => {
      const pid = 22222;
      const history = new Map();
      history.set(pid, {
        command: 'prompt_script.sh',
        status: 'exited',
        startTime: Date.now(),
      });
      (ShellExecutionService as any).backgroundProcessHistory.set(
        'default',
        history,
      );

      vi.spyOn(ExecutionLifecycleService, 'isActive').mockReturnValue(false);

      const invocation = writeTool.build({ pid, input: 'yes' });
      (invocation as any).context = { config: { getSessionId: () => 'default' } };
      const result = await invocation.execute({
        abortSignal: new AbortController().signal,
      });

      expect(result.error).toBeDefined();
      expect(result.llmContent).toContain('is not active');
    });

    it('should write input to active background process', async () => {
      const pid = 33333;
      const history = new Map();
      history.set(pid, {
        command: 'read_prompt.sh',
        status: 'running',
        startTime: Date.now(),
      });
      (ShellExecutionService as any).backgroundProcessHistory.set(
        'default',
        history,
      );

      vi.spyOn(ExecutionLifecycleService, 'isActive').mockReturnValue(true);
      const writeSpy = vi.spyOn(ExecutionLifecycleService, 'writeInput').mockImplementation(() => {});

      const invocation = writeTool.build({ pid, input: 'y' });
      (invocation as any).context = { config: { getSessionId: () => 'default' } };
      const result = await invocation.execute({
        abortSignal: new AbortController().signal,
      });

      expect(writeSpy).toHaveBeenCalledWith(pid, 'y\n');
      expect(result.llmContent).toContain('Successfully sent input to background process 33333');
    });
  });
});
