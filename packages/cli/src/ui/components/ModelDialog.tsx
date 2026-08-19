/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type React from 'react';
import { useCallback, useContext, useMemo, useState, useEffect } from 'react';
import { Box, Text } from 'ink';
import { ModelQuotaDisplay } from './ModelQuotaDisplay.js';
import { useUIState } from '../contexts/UIStateContext.js';
import {
  PREVIEW_GEMINI_MODEL,
  PREVIEW_GEMINI_3_1_MODEL,
  PREVIEW_GEMINI_FLASH_MODEL,
  PREVIEW_GEMINI_FLASH_LITE_MODEL,
  DEFAULT_GEMINI_MODEL,
  DEFAULT_GEMINI_FLASH_MODEL,
  DEFAULT_GEMINI_FLASH_LITE_MODEL,
  GEMINI_MODEL_ALIAS_AUTO,
  DEEPSEEK_CHAT_MODEL,
  DEEPSEEK_REASONER_MODEL,
  GEMMA_4_31B_IT_MODEL,
  GEMMA_4_26B_A4B_IT_MODEL,
  ModelSlashCommandEvent,
  logModelSlashCommand,
  getDisplayString,
  AuthType,
  PREVIEW_GEMINI_3_1_CUSTOM_TOOLS_MODEL,
  isProModel,
  getAutoModelDescription,
} from '@google/gemini-cli-core';
import { useKeypress } from '../hooks/useKeypress.js';
import { theme } from '../semantic-colors.js';
import { DescriptiveRadioButtonSelect } from './shared/DescriptiveRadioButtonSelect.js';
import { ConfigContext } from '../contexts/ConfigContext.js';
import { useSettings } from '../contexts/SettingsContext.js';

interface ModelDialogProps {
  onClose: () => void;
}

const DEEPSEEK_CONFIG_VALUE = '__deepseek_flash_config__';
const DEEPSEEK_PRO_CONFIG_VALUE = '__deepseek_pro_config__';

const REASONING_EFFORT_LEVELS: Array<'low' | 'medium' | 'high'> = ['low', 'medium', 'high'];
const REASONING_EFFORT_LABELS: Record<string, string> = {
  low: 'Fast & concise — simple tasks',
  medium: 'Balanced — most problems',
  high: 'Deep reasoning — complex architecture & hard bugs',
};
const REASONING_EFFORT_COLORS: Record<string, string> = {
  low: '#69f0ae',   // green  — light, fast
  medium: '#ff9800', // orange — moderate effort
  high: '#f44336',  // red    — maximum thinking
};

const PRO_REASONING_EFFORT_LEVELS: Array<'low' | 'medium' | 'high' | 'max'> = [
  'low',
  'medium',
  'high',
  'max',
];
const PRO_REASONING_EFFORT_LABELS: Record<string, string> = {
  low: 'Fast & concise — quick thinking',
  medium: 'Balanced — daily coding tasks',
  high: 'Deep reasoning — complex architecture & hard bugs (Standard)',
  max: 'Maximum depth — large scale refactoring & novel design',
};
const PRO_REASONING_EFFORT_COLORS: Record<string, string> = {
  low: '#69f0ae',   // green  — light, fast
  medium: '#ff9800', // orange — moderate effort
  high: '#f44336',  // red    — deep thinking
  max: '#e040fb',   // purple / magenta — maximum thinking power
};

const PRO_CONFIG_ROWS = ['reasoning', 'searchReasoning', 'persistence'] as const;
type ProConfigRow = typeof PRO_CONFIG_ROWS[number];

const TEMPERATURE_PRESETS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0];
const TEMPERATURE_LABELS: Record<number, string> = {
  0.0: 'Deterministic — exact reproducibility',
  0.1: 'Precise — best for code & bugs',
  0.2: 'Precise — best for code & bugs',
  0.3: 'Balanced — code + light creativity',
  0.5: 'Balanced — code + light creativity',
  0.7: 'Creative — docs & brainstorming',
  1.0: 'Default — natural conversation',
  1.2: 'Very creative — experimental',
  1.5: 'Very creative — experimental',
  2.0: 'Maximum randomness',
};

const CONFIG_ROWS = ['temperature', 'reasoning', 'searchReasoning', 'persistence'] as const;
type ConfigRow = typeof CONFIG_ROWS[number];

const SEARCH_REASONING_LEVELS: Array<'low' | 'medium' | 'high' | 'max'> = ['low', 'medium', 'high', 'max'];
const SEARCH_REASONING_LABELS: Record<string, string> = {
  low: 'Fast snippets & links (~2-4s)',
  medium: 'Balanced search & overview (~6-10s)',
  high: 'Deep multi-page crawl & full synthesis (~15-20s)',
  max: 'Exhaustive multi-source reasoning (~25-30s)',
};

export function getTempColor(temp: number): string {
  if (temp <= 0.2) return '#4fc3f7'; // Cyan / Blue (Cold, Precise, Deterministic)
  if (temp <= 0.5) return '#69f0ae'; // Green (Mild, Balanced)
  if (temp <= 1.0) return '#ffd54f'; // Warm Yellow (Default conversation)
  if (temp <= 1.5) return '#ff9800'; // Orange (Hot, Creative)
  return '#f44336';                  // Red (Maximum randomness / Fire)
}

function getTempLabel(temp: number): string {
  const closest = TEMPERATURE_PRESETS.reduce((prev, curr) =>
    Math.abs(curr - temp) < Math.abs(prev - temp) ? curr : prev,
  );
  return TEMPERATURE_LABELS[closest] ?? '';
}

export function ModelDialog({ onClose }: ModelDialogProps): React.JSX.Element {
  const config = useContext(ConfigContext);
  const settings = useSettings();
  const { terminalWidth } = useUIState();
  const [hasAccessToProModel, setHasAccessToProModel] = useState<boolean>(
    () => !(config?.getProModelNoAccessSync() ?? false),
  );
  const [view, setView] = useState<'main' | 'manual' | 'flash-config' | 'pro-config'>(() =>
    config?.getProModelNoAccessSync() ? 'manual' : 'main',
  );
  const [persistMode, setPersistMode] = useState(false);
  const [flashPersistMode, setFlashPersistMode] = useState(true);
  const [proPersistMode, setProPersistMode] = useState(true);

  // Currently focused row inside the flash-config and pro-config subviews
  const [configRow, setConfigRow] = useState<ConfigRow>('temperature');
  const [proConfigRow, setProConfigRow] = useState<ProConfigRow>('reasoning');

  // Temperature state (V4-Flash)
  const [temperature, setTemperatureState] = useState<number>(
    () => config?.getTemperature?.() ?? 1.0,
  );

  // Reasoning effort state (V4-Flash thinking mode)
  const [reasoningEffort, setReasoningEffortState] = useState<'low' | 'medium' | 'high'>(
    () => config?.getReasoningEffort?.() ?? 'medium',
  );

  // Reasoning effort state (V4-Pro thinking mode)
  const [proReasoningEffort, setProReasoningEffortState] = useState<'low' | 'medium' | 'high' | 'max'>(
    () => (config as any)?.getProReasoningEffort?.() ?? 'high',
  );

  // Search reasoning effort state (WebSearch engine)
  const [searchReasoning, setSearchReasoningState] = useState<'low' | 'medium' | 'high'>(
    () => config?.getSearchReasoningEffort?.() ?? 'low',
  );

  // Search reasoning effort state for Pro
  const [proSearchReasoning, setProSearchReasoningState] = useState<'low' | 'medium' | 'high' | 'max'>(
    () => (config as any)?.getProSearchReasoningEffort?.() ?? 'high',
  );

  useEffect(() => {
    async function checkAccess() {
      if (!config) return;
      const noAccess = await config.getProModelNoAccess();
      setHasAccessToProModel(!noAccess);
      if (noAccess) {
        setView('manual');
      }
    }
    void checkAccess()
  }, [config]);

  const preferredModel = config?.getModel() || GEMINI_MODEL_ALIAS_AUTO;

  const shouldShowPreviewModels = config?.getHasAccessToPreviewModel() ?? false;
  const useGemini31 = config?.getGemini31LaunchedSync?.() ?? false;
  const useGemini3_5Flash = config?.hasGemini35FlashGAAccess?.() ?? false;
  const selectedAuthType = settings.merged.security.auth.selectedType;
  const useCustomToolModel =
    useGemini31 && selectedAuthType === AuthType.USE_GEMINI;

  const isDeepSeekAuth =
    config?.getContentGeneratorConfig()?.authType === AuthType.USE_DEEPSEEK;

  useKeypress(
    (key) => {
      // --- pro-config subview ---
      if (view === 'pro-config') {
        if (key.name === 'escape') {
          setView('main');
          return true;
        }
        if (key.name === 'up') {
          setProConfigRow((prev) => {
            const idx = PRO_CONFIG_ROWS.indexOf(prev);
            return PRO_CONFIG_ROWS[Math.max(0, idx - 1)];
          });
          return true;
        }
        if (key.name === 'down') {
          setProConfigRow((prev) => {
            const idx = PRO_CONFIG_ROWS.indexOf(prev);
            return PRO_CONFIG_ROWS[Math.min(PRO_CONFIG_ROWS.length - 1, idx + 1)];
          });
          return true;
        }
        if (key.name === 'left' || key.name === 'right' || key.name === 'tab') {
          if (proConfigRow === 'persistence') {
            setProPersistMode((prev) => {
              const next = !prev;
              (config as any)?.saveProSettings?.(next);
              return next;
            });
            return true;
          }
        }
        if (key.name === 'left') {
          if (proConfigRow === 'reasoning') {
            setProReasoningEffortState((prev) => {
              const idx = PRO_REASONING_EFFORT_LEVELS.indexOf(prev);
              const next = PRO_REASONING_EFFORT_LEVELS[Math.max(0, idx - 1)];
              (config as any)?.setProReasoningEffort?.(next, proPersistMode);
              return next;
            });
          } else if (proConfigRow === 'searchReasoning') {
            setProSearchReasoningState((prev) => {
              const idx = PRO_REASONING_EFFORT_LEVELS.indexOf(prev);
              const next = PRO_REASONING_EFFORT_LEVELS[Math.max(0, idx - 1)];
              (config as any)?.setProSearchReasoningEffort?.(next, proPersistMode);
              return next;
            });
          }
          return true;
        }
        if (key.name === 'right') {
          if (proConfigRow === 'reasoning') {
            setProReasoningEffortState((prev) => {
              const idx = PRO_REASONING_EFFORT_LEVELS.indexOf(prev);
              const next = PRO_REASONING_EFFORT_LEVELS[Math.min(PRO_REASONING_EFFORT_LEVELS.length - 1, idx + 1)];
              (config as any)?.setProReasoningEffort?.(next, proPersistMode);
              return next;
            });
          } else if (proConfigRow === 'searchReasoning') {
            setProSearchReasoningState((prev) => {
              const idx = PRO_REASONING_EFFORT_LEVELS.indexOf(prev);
              const next = PRO_REASONING_EFFORT_LEVELS[Math.min(PRO_REASONING_EFFORT_LEVELS.length - 1, idx + 1)];
              (config as any)?.setProSearchReasoningEffort?.(next, proPersistMode);
              return next;
            });
          }
          return true;
        }
        return false;
      }

      // --- flash-config subview ---
      if (view === 'flash-config') {
        if (key.name === 'escape') {
          setView('main');
          return true;
        }
        if (key.name === 'up') {
          setConfigRow((prev) => {
            const idx = CONFIG_ROWS.indexOf(prev);
            return CONFIG_ROWS[Math.max(0, idx - 1)];
          });
          return true;
        }
        if (key.name === 'down') {
          setConfigRow((prev) => {
            const idx = CONFIG_ROWS.indexOf(prev);
            return CONFIG_ROWS[Math.min(CONFIG_ROWS.length - 1, idx + 1)];
          });
          return true;
        }
        if (key.name === 'left' || key.name === 'right' || key.name === 'tab') {
          if (configRow === 'persistence') {
            setFlashPersistMode((prev) => {
              const next = !prev;
              config?.saveFlashSettings?.(next);
              return next;
            });
            return true;
          }
        }
        if (key.name === 'left') {
          if (configRow === 'temperature') {
            setTemperatureState((prev) => {
              const idx = TEMPERATURE_PRESETS.indexOf(prev);
              const next = TEMPERATURE_PRESETS[Math.max(0, idx - 1)];
              config?.setTemperature?.(next, flashPersistMode);
              return next;
            });
          } else if (configRow === 'reasoning') {
            setReasoningEffortState((prev) => {
              const idx = REASONING_EFFORT_LEVELS.indexOf(prev);
              const next = REASONING_EFFORT_LEVELS[Math.max(0, idx - 1)];
              config?.setReasoningEffort?.(next, flashPersistMode);
              return next;
            });
          } else if (configRow === 'searchReasoning') {
            setSearchReasoningState((prev) => {
              const idx = SEARCH_REASONING_LEVELS.indexOf(prev);
              const next = SEARCH_REASONING_LEVELS[Math.max(0, idx - 1)] as 'low' | 'medium' | 'high';
              config?.setSearchReasoningEffort?.(next, flashPersistMode);
              return next;
            });
          }
          return true;
        }
        if (key.name === 'right') {
          if (configRow === 'temperature') {
            setTemperatureState((prev) => {
              const idx = TEMPERATURE_PRESETS.indexOf(prev);
              const next = TEMPERATURE_PRESETS[Math.min(TEMPERATURE_PRESETS.length - 1, idx + 1)];
              config?.setTemperature?.(next, flashPersistMode);
              return next;
            });
          } else if (configRow === 'reasoning') {
            setReasoningEffortState((prev) => {
              const idx = REASONING_EFFORT_LEVELS.indexOf(prev);
              const next = REASONING_EFFORT_LEVELS[Math.min(REASONING_EFFORT_LEVELS.length - 1, idx + 1)];
              config?.setReasoningEffort?.(next, flashPersistMode);
              return next;
            });
          } else if (configRow === 'searchReasoning') {
            setSearchReasoningState((prev) => {
              const idx = SEARCH_REASONING_LEVELS.indexOf(prev);
              const next = SEARCH_REASONING_LEVELS[Math.min(SEARCH_REASONING_LEVELS.length - 1, idx + 1)] as 'low' | 'medium' | 'high';
              config?.setSearchReasoningEffort?.(next, flashPersistMode);
              return next;
            });
          }
          return true;
        }
        return false;
      }

      // --- main / manual views ---
      if (key.name === 'escape') {
        if (view === 'manual' && hasAccessToProModel) {
          setView('main');
        } else {
          onClose();
        }
        return true;
      }
      if (key.name === 'tab') {
        setPersistMode((prev) => !prev);
        return true;
      }
      return false;
    },
    { isActive: true },
  );

  // manualModelSelected MUST be declared before mainOptions (which references it)
  const manualModelSelected = useMemo(() => {
    if (
      config?.getExperimentalDynamicModelConfiguration?.() === true &&
      config.getModelConfigService
    ) {
      const def = config
        .getModelConfigService()
        .getModelDefinition(preferredModel);
      return def && def.tier !== 'auto' && def.isVisible === true
        ? preferredModel
        : '';
    }

    const manualModels = [
      DEFAULT_GEMINI_MODEL,
      DEFAULT_GEMINI_FLASH_MODEL,
      DEFAULT_GEMINI_FLASH_LITE_MODEL,
      PREVIEW_GEMINI_MODEL,
      PREVIEW_GEMINI_3_1_MODEL,
      PREVIEW_GEMINI_3_1_CUSTOM_TOOLS_MODEL,
      PREVIEW_GEMINI_FLASH_LITE_MODEL,
      PREVIEW_GEMINI_FLASH_MODEL,
    ].filter((m) => m !== 'none');
    if (manualModels.includes(preferredModel)) {
      return preferredModel;
    }
    return '';
  }, [preferredModel, config]);

  const mainOptions = useMemo(() => {
    if (isDeepSeekAuth) {
      return [
        {
          value: DEEPSEEK_CHAT_MODEL,
          title: 'DeepSeek-V4-Flash',
          description: `Fast, efficient and cost-effective (64k context)  ·  t:${temperature.toFixed(1)}  r:${reasoningEffort}`,
          key: DEEPSEEK_CHAT_MODEL,
        },
        {
          value: DEEPSEEK_REASONER_MODEL,
          title: 'DeepSeek-V4-Pro (Thinking)',
          description: `Superior performance with deep reasoning (64k context)  ·  r:${proReasoningEffort}`,
          key: DEEPSEEK_REASONER_MODEL,
        },
        {
          value: DEEPSEEK_CONFIG_VALUE,
          title: 'Configure Flash Settings',
          description: 'Adjust temperature & reasoning effort for V4-Flash',
          key: DEEPSEEK_CONFIG_VALUE,
        },
        {
          value: DEEPSEEK_PRO_CONFIG_VALUE,
          title: 'Configure Pro Settings',
          description: 'Adjust reasoning effort (low / high / max) for V4-Pro',
          key: DEEPSEEK_PRO_CONFIG_VALUE,
        },
      ];
    }
    if (
      config?.getExperimentalDynamicModelConfiguration?.() === true &&
      config.getModelConfigService
    ) {
      const allOptions = config
        .getModelConfigService()
        .getAvailableModelOptions({
          useGemini3_1: useGemini31,
          useGemini3_5Flash,
          useCustomTools: useCustomToolModel,
          hasAccessToPreview: shouldShowPreviewModels,
          hasAccessToProModel,
        });

      const list = allOptions
        .filter((o) => o.tier === 'auto')
        .map((o) => ({
          value: o.modelId,
          title: o.name,
          description: o.description,
          key: o.modelId,
        }));

      list.push({
        value: 'Manual',
        title: manualModelSelected
          ? `Manual (${getDisplayString(manualModelSelected, config ?? undefined)})`
          : 'Manual',
        description: 'Manually select a model',
        key: 'Manual',
      });
      return list;
    }

    // --- LEGACY PATH ---
    const list = [
      {
        value: GEMINI_MODEL_ALIAS_AUTO,
        title: getDisplayString(GEMINI_MODEL_ALIAS_AUTO),
        description: getAutoModelDescription(
          shouldShowPreviewModels,
          useGemini31,
          useGemini3_5Flash,
        ),
        key: GEMINI_MODEL_ALIAS_AUTO,
      },
      {
        value: 'Manual',
        title: manualModelSelected
          ? `Manual (${getDisplayString(manualModelSelected)})`
          : 'Manual',
        description: 'Manually select a model',
        key: 'Manual',
      },
    ];

    return list;
  }, [
    config,
    isDeepSeekAuth,
    shouldShowPreviewModels,
    manualModelSelected,
    useGemini31,
    useGemini3_5Flash,
    useCustomToolModel,
    hasAccessToProModel,
    temperature,
    reasoningEffort,
    proReasoningEffort,
  ]);

  const manualOptions = useMemo(() => {
    if (
      config?.getExperimentalDynamicModelConfiguration?.() === true &&
      config.getModelConfigService
    ) {
      const allOptions = config
        .getModelConfigService()
        .getAvailableModelOptions({
          useGemini3_1: useGemini31,
          useGemini3_5Flash,
          useCustomTools: useCustomToolModel,
          hasAccessToPreview: shouldShowPreviewModels,
          hasAccessToProModel,
        });

      return allOptions
        .filter((o) => o.tier !== 'auto')
        .map((o) => ({
          value: o.modelId,
          title: o.name,
          key: o.modelId,
        }));
    }

    const showGemmaModels = config?.getExperimentalGemma() ?? false;

    const options = [
      {
        value: DEFAULT_GEMINI_MODEL,
        title: getDisplayString(DEFAULT_GEMINI_MODEL),
        key: DEFAULT_GEMINI_MODEL,
      },
      {
        value: DEFAULT_GEMINI_FLASH_LITE_MODEL,
        title: getDisplayString(DEFAULT_GEMINI_FLASH_LITE_MODEL),
        key: DEFAULT_GEMINI_FLASH_LITE_MODEL,
      },
      {
        value: DEFAULT_GEMINI_FLASH_MODEL,
        title: getDisplayString(DEFAULT_GEMINI_FLASH_MODEL),
        key: DEFAULT_GEMINI_FLASH_MODEL,
      },
    ];

    if (showGemmaModels) {
      options.push(
        {
          value: GEMMA_4_31B_IT_MODEL,
          title: getDisplayString(GEMMA_4_31B_IT_MODEL),
          key: GEMMA_4_31B_IT_MODEL,
        },
        {
          value: GEMMA_4_26B_A4B_IT_MODEL,
          title: getDisplayString(GEMMA_4_26B_A4B_IT_MODEL),
          key: GEMMA_4_26B_A4B_IT_MODEL,
        },
      );
    }

    if (shouldShowPreviewModels) {
      const previewProModel = useGemini31
        ? PREVIEW_GEMINI_3_1_MODEL
        : PREVIEW_GEMINI_MODEL;

      const previewProValue = useCustomToolModel
        ? PREVIEW_GEMINI_3_1_CUSTOM_TOOLS_MODEL
        : previewProModel;

      const previewOptions = [
        {
          value: previewProValue,
          title: getDisplayString(previewProModel),
          key: previewProModel,
        },
        {
          value: PREVIEW_GEMINI_FLASH_MODEL,
          title: getDisplayString(PREVIEW_GEMINI_FLASH_MODEL),
          key: PREVIEW_GEMINI_FLASH_MODEL,
        },
      ];

      if (PREVIEW_GEMINI_FLASH_LITE_MODEL !== 'none') {
        previewOptions.push({
          value: PREVIEW_GEMINI_FLASH_LITE_MODEL,
          title: getDisplayString(PREVIEW_GEMINI_FLASH_LITE_MODEL),
          key: PREVIEW_GEMINI_FLASH_LITE_MODEL,
        });
      }

      options.unshift(...previewOptions);
    }

    if (!hasAccessToProModel) {
      return options.filter((option) => !isProModel(option.value));
    }

    return options;
  }, [
    shouldShowPreviewModels,
    useGemini31,
    useGemini3_5Flash,
    useCustomToolModel,
    hasAccessToProModel,
    config,
  ]);

  const options = useMemo(() => {
    const rawOptions = view === 'main' ? mainOptions : manualOptions;
    const seen = new Set<string>();
    return rawOptions.filter((option) => {
      if (seen.has(option.value)) return false;
      seen.add(option.value);
      return true;
    });
  }, [view, mainOptions, manualOptions]);

  const initialIndex = useMemo(() => {
    const idx = options.findIndex((option) => option.value === preferredModel);
    if (idx !== -1) return idx;
    if (view === 'main') {
      const manualIdx = options.findIndex((o) => o.value === 'Manual');
      return manualIdx !== -1 ? manualIdx : 0;
    }
    return 0;
  }, [preferredModel, options, view]);

  const handleSelect = useCallback(
    (model: string) => {
      if (model === DEEPSEEK_CONFIG_VALUE) {
        setView('flash-config');
        return;
      }
      if (model === DEEPSEEK_PRO_CONFIG_VALUE) {
        setView('pro-config');
        return;
      }
      if (model === 'Manual') {
        setView('manual');
        return;
      }
      if (config) {
        config.setModel(model, persistMode ? false : true);
        const event = new ModelSlashCommandEvent(model);
        logModelSlashCommand(config, event);
      }
      onClose();
    },
    [config, onClose, persistMode],
  );

  // ──────────────────────────────────────────────
  // Pro Config subview
  // ──────────────────────────────────────────────
  if (view === 'pro-config') {
    const effortIdx = PRO_REASONING_EFFORT_LEVELS.indexOf(proReasoningEffort);
    const canEffortLeft = effortIdx > 0;
    const canEffortRight = effortIdx < PRO_REASONING_EFFORT_LEVELS.length - 1;
    const effortColor = PRO_REASONING_EFFORT_COLORS[proReasoningEffort] ?? '#f44336';

    const searchEffortIdx = PRO_REASONING_EFFORT_LEVELS.indexOf(proSearchReasoning);
    const canSearchEffortLeft = searchEffortIdx > 0;
    const canSearchEffortRight = searchEffortIdx < PRO_REASONING_EFFORT_LEVELS.length - 1;
    const searchEffortColor = PRO_REASONING_EFFORT_COLORS[proSearchReasoning] ?? '#f44336';

    const isEffortFocused = proConfigRow === 'reasoning';
    const isSearchFocused = proConfigRow === 'searchReasoning';
    const isPersistFocused = proConfigRow === 'persistence';

    return (
      <Box
        borderStyle="round"
        borderColor={theme.border.default}
        flexDirection="column"
        padding={1}
        width="100%"
      >
        <Text bold>Configure Pro Settings (DeepSeek-V4-Pro)</Text>
        <Text color={theme.text.secondary}>Use up/down to switch row · left/right to change value · Esc to go back</Text>

        <Box flexDirection="column" marginTop={1}>
          {/* Model reasoning effort row */}
          <Box>
            <Text color={isEffortFocused ? theme.text.accent : theme.text.secondary}>
              {isEffortFocused ? '▶ ' : '  '}
            </Text>
            <Text bold color={isEffortFocused ? theme.text.primary : theme.text.secondary}>
              {'Model Reasoning:   '}
            </Text>
            <Text color={canEffortLeft && isEffortFocused ? theme.text.accent : theme.text.secondary}>◄ </Text>
            <Text bold color={isEffortFocused ? effortColor : theme.text.secondary}>
              {proReasoningEffort.toUpperCase()}
            </Text>
            <Text color={canEffortRight && isEffortFocused ? theme.text.accent : theme.text.secondary}> ►</Text>
            <Text color={effortColor}>  {PRO_REASONING_EFFORT_LABELS[proReasoningEffort]}</Text>
          </Box>

          {/* Search reasoning effort row */}
          <Box marginTop={0}>
            <Text color={isSearchFocused ? theme.text.accent : theme.text.secondary}>
              {isSearchFocused ? '▶ ' : '  '}
            </Text>
            <Text bold color={isSearchFocused ? theme.text.primary : theme.text.secondary}>
              {'Search Reasoning:  '}
            </Text>
            <Text color={canSearchEffortLeft && isSearchFocused ? theme.text.accent : theme.text.secondary}>◄ </Text>
            <Text bold color={isSearchFocused ? searchEffortColor : theme.text.secondary}>
              {proSearchReasoning.toUpperCase()}
            </Text>
            <Text color={canSearchEffortRight && isSearchFocused ? theme.text.accent : theme.text.secondary}> ►</Text>
            <Text color={searchEffortColor}>  {SEARCH_REASONING_LABELS[proSearchReasoning] ?? ''}</Text>
          </Box>

          {/* Persistence row */}
          <Box marginTop={0}>
            <Text color={isPersistFocused ? theme.text.accent : theme.text.secondary}>
              {isPersistFocused ? '▶ ' : '  '}
            </Text>
            <Text bold color={isPersistFocused ? theme.text.primary : theme.text.secondary}>
              {'Persistence:       '}
            </Text>
            <Text color={isPersistFocused ? theme.text.accent : theme.text.secondary}>◄ </Text>
            <Text bold color={isPersistFocused ? (proPersistMode ? '#69f0ae' : '#ff9800') : theme.text.secondary}>
              {proPersistMode ? 'PERMANENT' : 'SESSION ONLY'}
            </Text>
            <Text color={isPersistFocused ? theme.text.accent : theme.text.secondary}> ►</Text>
            <Text color={isPersistFocused ? (proPersistMode ? '#69f0ae' : '#ff9800') : theme.text.secondary}>
              {proPersistMode
                ? '  Saved to disk (~/.deepseek/pro_settings.json)'
                : '  Active in this conversation (resets on restart)'}
            </Text>
          </Box>
        </Box>

        <Box marginTop={1}>
          <Text color={theme.text.secondary}>(Press Esc to go back)</Text>
        </Box>
      </Box>
    );
  }

  // ──────────────────────────────────────────────
  // Flash Config subview
  // ──────────────────────────────────────────────
  if (view === 'flash-config') {
    const tempIdx = TEMPERATURE_PRESETS.indexOf(temperature);
    const canTempLeft = tempIdx > 0;
    const canTempRight = tempIdx < TEMPERATURE_PRESETS.length - 1;

    const effortIdx = REASONING_EFFORT_LEVELS.indexOf(reasoningEffort);
    const canEffortLeft = effortIdx > 0;
    const canEffortRight = effortIdx < REASONING_EFFORT_LEVELS.length - 1;
    const effortColor = REASONING_EFFORT_COLORS[reasoningEffort] ?? '#69f0ae';

    const searchEffortIdx = SEARCH_REASONING_LEVELS.indexOf(searchReasoning);
    const canSearchEffortLeft = searchEffortIdx > 0;
    const canSearchEffortRight = searchEffortIdx < SEARCH_REASONING_LEVELS.length - 1;
    const searchEffortColor = REASONING_EFFORT_COLORS[searchReasoning] ?? '#69f0ae';

    const isTempFocused = configRow === 'temperature';
    const isEffortFocused = configRow === 'reasoning';
    const isSearchFocused = configRow === 'searchReasoning';
    const isPersistFocused = configRow === 'persistence';

    return (
      <Box
        borderStyle="round"
        borderColor={theme.border.default}
        flexDirection="column"
        padding={1}
        width="100%"
      >
        <Text bold>Configure Flash Settings</Text>
        <Text color={theme.text.secondary}>Use up/down to switch row · left/right to change value · Esc to go back</Text>

        <Box flexDirection="column" marginTop={1}>
          {/* Temperature row */}
          <Box>
            <Text color={isTempFocused ? theme.text.accent : theme.text.secondary}>
              {isTempFocused ? '▶ ' : '  '}
            </Text>
            <Text bold color={isTempFocused ? theme.text.primary : theme.text.secondary}>
              {'Temperature:       '}
            </Text>
            <Text color={canTempLeft && isTempFocused ? theme.text.accent : theme.text.secondary}>◄ </Text>
            <Text bold color={isTempFocused ? getTempColor(temperature) : theme.text.secondary}>
              {temperature.toFixed(1)}
            </Text>
            <Text color={canTempRight && isTempFocused ? theme.text.accent : theme.text.secondary}> ►</Text>
            <Text color={isTempFocused ? getTempColor(temperature) : theme.text.secondary}>
              {'  '}{getTempLabel(temperature)}
            </Text>
          </Box>

          {/* Model reasoning effort row */}
          <Box marginTop={0}>
            <Text color={isEffortFocused ? theme.text.accent : theme.text.secondary}>
              {isEffortFocused ? '▶ ' : '  '}
            </Text>
            <Text bold color={isEffortFocused ? theme.text.primary : theme.text.secondary}>
              {'Model Reasoning:   '}
            </Text>
            <Text color={canEffortLeft && isEffortFocused ? theme.text.accent : theme.text.secondary}>◄ </Text>
            <Text bold color={isEffortFocused ? effortColor : theme.text.secondary}>
              {reasoningEffort.toUpperCase()}
            </Text>
            <Text color={canEffortRight && isEffortFocused ? theme.text.accent : theme.text.secondary}> ►</Text>
            <Text color={effortColor}>  {REASONING_EFFORT_LABELS[reasoningEffort]}</Text>
          </Box>

          {/* Search reasoning effort row */}
          <Box marginTop={0}>
            <Text color={isSearchFocused ? theme.text.accent : theme.text.secondary}>
              {isSearchFocused ? '▶ ' : '  '}
            </Text>
            <Text bold color={isSearchFocused ? theme.text.primary : theme.text.secondary}>
              {'Search Reasoning:  '}
            </Text>
            <Text color={canSearchEffortLeft && isSearchFocused ? theme.text.accent : theme.text.secondary}>◄ </Text>
            <Text bold color={isSearchFocused ? searchEffortColor : theme.text.secondary}>
              {searchReasoning.toUpperCase()}
            </Text>
            <Text color={canSearchEffortRight && isSearchFocused ? theme.text.accent : theme.text.secondary}> ►</Text>
            <Text color={searchEffortColor}>  {SEARCH_REASONING_LABELS[searchReasoning]}</Text>
          </Box>

          {/* Persistence row */}
          <Box marginTop={0}>
            <Text color={isPersistFocused ? theme.text.accent : theme.text.secondary}>
              {isPersistFocused ? '▶ ' : '  '}
            </Text>
            <Text bold color={isPersistFocused ? theme.text.primary : theme.text.secondary}>
              {'Persistence:       '}
            </Text>
            <Text color={isPersistFocused ? theme.text.accent : theme.text.secondary}>◄ </Text>
            <Text bold color={isPersistFocused ? (flashPersistMode ? '#69f0ae' : '#ff9800') : theme.text.secondary}>
              {flashPersistMode ? 'PERMANENT' : 'SESSION ONLY'}
            </Text>
            <Text color={isPersistFocused ? theme.text.accent : theme.text.secondary}> ►</Text>
            <Text color={isPersistFocused ? (flashPersistMode ? '#69f0ae' : '#ff9800') : theme.text.secondary}>
              {flashPersistMode
                ? '  Saved to disk (~/.deepseek/flash_settings.json)'
                : '  Active in this conversation (resets on restart)'}
            </Text>
          </Box>
        </Box>

        <Box marginTop={1}>
          <Text color={theme.text.secondary}>(Press Esc to go back)</Text>
        </Box>
      </Box>
    );
  }

  // ──────────────────────────────────────────────
  // Main / Manual view
  // ──────────────────────────────────────────────
  return (
    <Box
      borderStyle="round"
      borderColor={theme.border.default}
      flexDirection="column"
      padding={1}
      width="100%"
    >
      <Text bold>Select Model</Text>

      <Box marginTop={1}>
        <DescriptiveRadioButtonSelect
          items={options}
          onSelect={handleSelect}
          initialIndex={initialIndex}
          showNumbers={true}
        />
      </Box>

      <Box marginTop={1} flexDirection="column">
        <Box>
          <Text bold color={theme.text.primary}>
            Remember model for future sessions:{' '}
          </Text>
          <Text color={theme.status.success}>
            {persistMode ? 'true' : 'false'}
          </Text>
          <Text color={theme.text.secondary}> (Press Tab to toggle)</Text>
        </Box>
      </Box>
      {!isDeepSeekAuth && (
        <Box flexDirection="column">
          <Text color={theme.text.secondary}>
            {'> To use a specific Gemini model on startup, use the --model flag.'}
          </Text>
        </Box>
      )}
      <ModelQuotaDisplay
        buckets={config?.getLastRetrievedQuota()?.buckets}
        availableWidth={terminalWidth - 2}
      />
      <Box marginTop={1} flexDirection="column">
        <Text color={theme.text.secondary}>
          (Press Enter or 1-4 to select · Tab for persistence · Esc to cancel)
        </Text>
      </Box>
    </Box>
  );
}
