"""
EX0S-DEV1 ModularOrganism public API.

The organism is the unit of behavioral evaluation. It is born, lives,
and dies in a single continuous life. The runner calls only the public
API defined here.

Public API
──────────
    birth(genome) → ModularOrganism
    observe(event) → OrganismObservation
    act() → ActionResult
    rest(n_ticks) → dict   (rest telemetry)
    episode_reset() → EpisodeReset
    full_checkpoint() → FullCheckpoint
    hippocampal_graft(graft: HippocampalGraft) → None

Within-life learning law (enforced here; audited in test_boundaries.py)
──────────────────────────────────────────────────────────────────────
Allowed:
- Sensory events and temporal order
- Organism's chosen motor action and efference copy
- Teacher-demonstrated action through the same motor-observation channel
- Scalar reward/advantage as a gate (not an answer identifier)
- Internally generated prediction error, novelty, conflict, eligibility traces
- Replay sampled by organism from H

Forbidden:
- Expected action handles in weight updates
- Cue IDs / logical slots / fixture metadata
- Runner-generated keys, queries, stored values, or retrieval addresses
- Future probes or validation worlds
- Oracle W*; LLM; external dictionary; planted semantic vector

NO GRADIENT-BASED UPDATE is applied to W, H, or ρ during an organism's
evaluated lifetime. Within-life learning uses ONLY the frozen local
plasticity law. The research optimizer may differentiate through completed
training-life trajectories, but it may update ONLY inherited G between
lives. Validation and confirmation lives never contribute gradients.
"""

from __future__ import annotations

import random
import time
import copy

import torch
import torch.nn.functional as F
import numpy as np

from three_memory.dev1.genome import DevGenome
from three_memory.dev1.interfaces import (
    ActionResult,
    EpisodeReset,
    FullCheckpoint,
    HippocampalGraft,
    OrganismObservation,
    OrganismTelemetry,
)
from three_memory.dev1.cortex import SensoryCortex, RelationalCortex, ActionCortex
from three_memory.dev1.hippocampus import FastHippocampus
from three_memory.dev1.working_state import WorkingState
from three_memory.dev1.neuromod import NeuromodController, EligibilityTrace
from three_memory.dev1.provenance import ProvenanceLog, EventKind

from three_memory.dev1.plasticity.cortical_plasticity.three_factor import RewardBaselineThreeFactor
from three_memory.dev1.plasticity.cortical_plasticity.meta_learned import ActionContingentActorCritic
from three_memory.dev1.plasticity.cortical_plasticity.evolved import ConsequencePredictionCredit
from three_memory.dev1.plasticity.eprop.reward_eprop import RewardEpropRateAdaptation
from three_memory.dev1.plasticity.eprop.learned_eprop import InheritedSignalGeneratorEprop
from three_memory.dev1.plasticity.consolidation.replay import ReplayConsolidation
from three_memory.dev1.plasticity.consolidation.online_slow import OnlineSlowConsolidation
from three_memory.dev1.plasticity.consolidation.hybrid import HybridConsolidation


def _build_cortical_plasticity(
    family: str,
    genome: DevGenome,
    n_pre: int,
    n_post: int,
    device: torch.device | None = None,
):
    dev = device or torch.device("cpu")
    if family == "reward_baseline_three_factor":
        return RewardBaselineThreeFactor(genome.plasticity)
    elif family == "action_contingent_actor_critic":
        return ActionContingentActorCritic(n_pre, n_post, genome.plasticity)
    elif family == "consequence_prediction_credit":
        return ConsequencePredictionCredit(n_pre, n_post, genome.plasticity)
    elif family in ("reward_eprop_rate_adaptation", "teacher_demo_eprop", "r2_fixed_eprop_baseline"):
        return RewardEpropRateAdaptation(genome, n_pre, n_post, dev)
    elif family == "inherited_learning_signal_generator":
        return InheritedSignalGeneratorEprop(genome, n_pre, n_post, dev)
    else:
        raise ValueError(f"Unknown plasticity_family: {family!r}")


def _build_consolidation(family: str, genome: DevGenome):
    if family == "replay":
        return ReplayConsolidation(genome)
    elif family == "online_slow":
        return OnlineSlowConsolidation(genome)
    elif family == "hybrid":
        return HybridConsolidation(genome)
    else:
        raise ValueError(f"Unknown consolidation_family: {family!r}")


class ModularOrganism:
    """
    One newborn organism.

    Do not instantiate directly — use ModularOrganism.birth(genome).
    """

    def __init__(
        self,
        genome: DevGenome,
        device: torch.device | None = None,
        consolidation_disabled: bool = False,
        h_disabled: bool = False,
    ):
        self.genome = genome
        self.device = device or torch.device("cpu")
        self.consolidation_disabled = consolidation_disabled
        self.step: int = 0
        self.age_frac: float = 0.0
        self._max_steps_hint: int = 10_000   # updated by runner if known

        # Cortical populations
        self.sensory_ctx = SensoryCortex(genome, self.device)
        self.relational_ctx = RelationalCortex(genome, self.device)
        self.action_ctx = ActionCortex(genome, self.device)

        # Fast memory
        self.hippocampus = FastHippocampus(genome, self.device)
        self.hippocampus.h_disabled = h_disabled

        # Working state (transient)
        self.rho = WorkingState(genome, self.device)

        # Neuromodulation
        self.neuromod = NeuromodController(genome, self.device)
        self.eligibility = EligibilityTrace(
            genome.relational_ctx.n_units,
            genome.action_ctx.n_units,
            genome.plasticity.eligibility_decay,
            self.device,
        )

        # Cortical plasticity rule
        self.plasticity_rule = _build_cortical_plasticity(
            genome.plasticity_family,
            genome,
            genome.relational_ctx.n_units,
            genome.action_ctx.n_units,
            device=self.device,
        )
        self._last_consequence_reward: float = 0.0
        self._last_consequence_terminal: bool = False
        self._last_motor_logits: torch.Tensor | None = None

        # Consolidation
        self.consolidation = _build_consolidation(genome.consolidation_family, genome)

        # Provenance log
        self.slog = ProvenanceLog()
        self.slog.append(EventKind.BIRTH, step=0, payload={"genome_hash": genome.genome_hash()})

        # Internal surprises (for prioritized replay)
        self._surprises: list[float] = []

        # Outcome-credit lifecycle (R2.1) — self-action only
        self._awaiting_consequence: bool = False
        self._outcome_credit_pending: bool = False
        self._outcome_credit_consumed: bool = False
        self._credit_interaction_step: int = -1

        # Teacher-demonstration credit lifecycle (Reference Birth R2) — separate from self
        self._teacher_awaiting_consequence: bool = False
        self._teacher_credit_pending: bool = False
        self._teacher_credit_consumed: bool = False
        self._teacher_demo_channel: int | None = None
        self._teacher_teaching_signal: float = 0.0
        self._teacher_eligibility: torch.Tensor | None = None
        self._teacher_credit_enabled: bool = True

    @classmethod
    def birth(
        cls,
        genome: DevGenome,
        device: torch.device | None = None,
        consolidation_disabled: bool = False,
        h_disabled: bool = False,
    ) -> "ModularOrganism":
        """
        Produce a newborn organism from genome G.
        W, H, ρ, S_log all begin empty/newborn.
        """
        torch.manual_seed(genome.seed)
        random.seed(genome.seed)
        return cls(genome, device=device, consolidation_disabled=consolidation_disabled, h_disabled=h_disabled)

    # ── Public API ─────────────────────────────────────────────────────────────

    def observe(self, event: OrganismObservation) -> OrganismObservation:
        """
        Process one sensory event. Updates working state via cortical forward pass.
        No gradient update to W, H, or ρ.
        """
        sensory_v = _to_tensor(event.sensory_vector, self.device)
        sensory_v = _pad_or_trim(sensory_v, self.genome.sensory_dim)

        # Forward pass
        self.rho.sensory_repr = self.sensory_ctx(sensory_v, self.rho.sensory_repr)

        if self.hippocampus.h_disabled:
            retrieved = torch.zeros(self.genome.relational_ctx.n_units, device=self.device)
        else:
            retrieved = self.hippocampus.read(self.rho.relational_repr)
        self.rho.relational_repr = self.relational_ctx(
            self.rho.sensory_repr, retrieved, self.rho.relational_repr
        )

        self.rho.step = self.step
        self.step += 1
        self.age_frac = min(1.0, self.step / max(1, self._max_steps_hint))

        # Neuromodulatory signals (no W update here)
        mod = self.neuromod(
            self.rho.sensory_repr,
            self.rho.relational_repr,
            self.rho.action_repr,
            event.reward,
        )
        self._last_mod = mod

        if self._awaiting_consequence:
            self._outcome_credit_pending = True
            self._outcome_credit_consumed = False
            self._awaiting_consequence = False
            self._last_consequence_reward = float(event.reward)
            self._last_consequence_terminal = bool(event.is_terminal)

        if event.observed_motor_event is not None:
            self._arm_teacher_demonstration(int(event.observed_motor_event))

        # Teacher consequence is never the self-action reward field.
        if event.teaching_signal is not None and self._teacher_awaiting_consequence:
            self._teacher_teaching_signal = float(event.teaching_signal)
            self._teacher_credit_pending = True
            self._teacher_credit_consumed = False
            self._teacher_awaiting_consequence = False

        # Eligibility update deferred to act() so action_repr is populated

        # Write to H if past onset age and H not disabled
        onset = self.genome.schedule.replay_onset_age_frac
        novelty_val = float(mod["novelty"].item())
        if (
            not self.hippocampus.h_disabled
            and self.age_frac >= onset
            and novelty_val > self.genome.schedule.memory_write_threshold
        ):
            self.hippocampus.write(self.rho.relational_repr, self.rho.sensory_repr)
            self._surprises.append(float(mod["prediction_error"].abs().item()))
            self.slog.append(EventKind.WRITE_H, step=self.step, payload={"novelty": novelty_val})

        self.slog.append(EventKind.OBSERVE, step=self.step, payload={
            "reward": event.reward,
            "terminal": event.is_terminal,
        })

        return event

    def act(
        self,
        policy_mode: str = "hard",
        action_generator: torch.Generator | None = None,
    ) -> ActionResult:
        """
        Produce motor output via action cortex competition.
        No gradient update to W, H, or ρ.
        """
        action_state, motor_logits = self.action_ctx(self.rho.relational_repr, self.rho.action_repr)
        if hasattr(self, "_r2_motor_channel_bias"):
            motor_logits = motor_logits + self._r2_motor_channel_bias
        self.rho.action_repr = action_state

        # Update eligibility trace here (after action state is populated)
        self.eligibility.update(self.rho.relational_repr, self.rho.action_repr)

        channel, scores, confidence = self.action_ctx.competition(
            motor_logits,
            policy_mode=policy_mode,
            generator=action_generator,
        )
        self._last_motor_logits = motor_logits.detach()
        self._last_action_channel = channel
        self._last_action_policy_mode = policy_mode
        self._last_action_log_prob = float(torch.log(scores[channel] + 1e-12).item())
        self._last_action_entropy = float(-(scores * torch.log(scores + 1e-12)).sum().item())
        self._awaiting_consequence = True
        self._outcome_credit_pending = False
        self._outcome_credit_consumed = False
        self._credit_interaction_step = self.step

        self.slog.append(EventKind.ACT, step=self.step, payload={
            "motor_channel": channel,
            "confidence": confidence,
            "policy_mode": policy_mode,
        })
        return ActionResult(
            motor_channel=channel,
            motor_scores=scores.detach().cpu().numpy(),
            confidence=confidence,
        )

    def _apply_local_plasticity(self) -> torch.Tensor | None:
        """
        Apply the within-life local plasticity rule to the action cortex W_motor.
        This is the ONLY weight update during an evaluated life.
        Uses no_grad() — this is not backpropagation.

        Allowed signals depend on family but are always local:
        eligibility trace, scalar reward-derived modulation, consequence error,
        and the organism's chosen motor channel.
        """
        if not hasattr(self, "_last_mod") or not hasattr(self, "_last_action_channel"):
            return None
        mod = self._last_mod
        rule = self.plasticity_rule
        elig = self.eligibility.trace   # shape (n_rel, n_action)
        chosen_channel = self._last_action_channel
        n_channels = self.genome.n_motor_channels

        with torch.no_grad():
            if rule.name() == "reward_baseline_three_factor":
                dW = rule.actor_delta(
                    eligibility=elig,
                    reward_baseline_error=mod["reward_baseline_error"],
                    chosen_channel=chosen_channel,
                    n_channels=n_channels,
                )
            elif rule.name() == "action_contingent_actor_critic":
                dW = rule.actor_delta(
                    eligibility=elig,
                    td_error=mod["td_error"],
                    chosen_channel=chosen_channel,
                    n_channels=n_channels,
                )
            elif rule.name() == "consequence_prediction_credit":
                dW = rule.actor_delta(
                    eligibility=elig,
                    reward_gate=mod["reward_gate"],
                    consequence_error=mod["consequence_error"],
                    chosen_channel=chosen_channel,
                    n_channels=n_channels,
                )
            elif rule.name() in (
                "reward_eprop_rate_adaptation",
                "inherited_learning_signal_generator",
            ):
                motor_logits = getattr(self, "_last_motor_logits", None)
                if motor_logits is None:
                    return None
                delta_t = rule.td_step(
                    self._last_consequence_reward,
                    self.rho.relational_repr,
                    self._last_consequence_terminal,
                )
                rule.update_critic(delta_t, self.rho.relational_repr)
                actor_kwargs = dict(
                    eligibility=elig,
                    delta_t=delta_t,
                    chosen_channel=chosen_channel,
                    motor_logits=motor_logits,
                    n_channels=n_channels,
                )
                if rule.name() == "inherited_learning_signal_generator":
                    actor_kwargs["relational_state"] = self.rho.relational_repr
                dW = rule.actor_delta(**actor_kwargs)
                if rule.name() == "inherited_learning_signal_generator":
                    self._last_learning_signal_norm = float(
                        rule.learning_signal_per_unit(
                            delta_t,
                            chosen_channel,
                            motor_logits,
                            relational_state=self.rho.relational_repr,
                        ).norm().item()
                    )
                else:
                    self._last_learning_signal_norm = float(
                        rule.learning_signal_per_unit(delta_t, chosen_channel, motor_logits).norm().item()
                    )
                self._last_td_error = float(delta_t.item())
                self._last_critic_value = float(rule.critic.value(self.rho.relational_repr).item())
            else:
                raise ValueError(f"Unknown local plasticity family: {rule.name()}")
            if hasattr(self, "_r2_plasticity_channel_mask"):
                dW = dW * self._r2_plasticity_channel_mask
            if hasattr(self, "_r2_plasticity_mask_gain"):
                dW = dW * float(self._r2_plasticity_mask_gain)
            self.action_ctx.W_motor.weight.data.add_(dW)
            self._last_actor_delta = dW.detach().clone()
        return dW

    def apply_outcome_credit(self) -> dict:
        """
        Apply eligible local plasticity immediately after outcome delivery.

        Stage A R2.1 credit lifecycle: observe → act → consequence → credit.
        Eligibility must be finite and nonzero before this call; reset must occur
        only after credit has been applied for the triggering interaction.

        Returns applied/refused status. A second call for the same interaction
        is refused. Checkpoint restore invalidates pending credit (deterministic rule).
        """
        if self._outcome_credit_consumed:
            return {
                "applied": False,
                "refused": True,
                "reason": "outcome_credit_already_consumed",
                "eligibility_norm_before_credit": 0.0,
                "rewarded_update_norm": 0.0,
                "signed_reward_projection": 0.0,
            }
        if not self._outcome_credit_pending:
            return {
                "applied": False,
                "refused": True,
                "reason": "outcome_credit_not_pending",
                "eligibility_norm_before_credit": 0.0,
                "rewarded_update_norm": 0.0,
                "signed_reward_projection": 0.0,
            }

        elig = self.eligibility.trace
        elig_norm = float(elig.norm().item())
        w_before = self.action_ctx.W_motor.weight.data.clone()

        rewarded_channel = getattr(self, "_last_action_channel", 0)
        # If reward was positive, the rewarded channel is the chosen one; margin is vs that channel.
        pre_stats = self._rewarded_action_stats(rewarded_channel)

        dW = self._apply_local_plasticity()
        if dW is None:
            dW = torch.zeros_like(w_before)
        update_norm = float(dW.norm().item())
        post_stats = self._rewarded_action_stats(rewarded_channel)
        margin_delta = post_stats["margin"] - pre_stats["margin"]
        mod = getattr(self, "_last_mod", {})
        reward_signal = 0.0
        if "reward_baseline_error" in mod:
            val = mod["reward_baseline_error"]
            reward_signal = float(val.item() if hasattr(val, "item") else val)
        elif "td_error" in mod:
            val = mod["td_error"]
            reward_signal = float(val.item() if hasattr(val, "item") else val)
        elif "reward_gate" in mod:
            val = mod["reward_gate"]
            reward_signal = float(val.item() if hasattr(val, "item") else val)
        elif hasattr(self, "_last_td_error"):
            reward_signal = float(self._last_td_error)
        chosen = getattr(self, "_last_action_channel", 0)
        channel_update = float(dW[chosen].norm().item()) if dW.numel() else 0.0
        signed_projection = reward_signal * channel_update

        self._outcome_credit_consumed = True
        self._outcome_credit_pending = False

        result = {
            "applied": True,
            "refused": False,
            "reason": "",
            "credit_source": "self_action",
            "actor_target": int(rewarded_channel),
            "eligibility_norm_before_credit": elig_norm,
            "rewarded_update_norm": update_norm,
            "signed_reward_projection": signed_projection,
            "pre_rewarded_action_probability": pre_stats["probability"],
            "post_rewarded_action_probability": post_stats["probability"],
            "pre_rewarded_action_margin": pre_stats["margin"],
            "post_rewarded_action_margin": post_stats["margin"],
            "margin_change_sign": int((margin_delta > 0) - (margin_delta < 0)),
            "learning_signal_norm": float(getattr(self, "_last_learning_signal_norm", 0.0)),
            "critic_value": float(getattr(self, "_last_critic_value", 0.0)),
            "td_error": float(getattr(self, "_last_td_error", reward_signal)),
            "action_entropy": float(getattr(self, "_last_action_entropy", 0.0)),
            "outcome_sign": int((self._last_consequence_reward > 0) - (self._last_consequence_reward < 0)),
        }
        self._last_outcome_credit = dict(result)
        return result

    def set_teacher_credit_enabled(self, enabled: bool) -> None:
        """Control: teacher credit off → state injection alone does not teach."""
        self._teacher_credit_enabled = bool(enabled)

    def apply_teacher_demonstration_credit(self) -> dict:
        """
        Apply credit for a teacher demonstration on its own eligibility/credit source.

        Actor target is the demonstrated channel. Learning signal is the teaching
        signal delivered with that demonstration — never the self-action reward.
        Does not rewrite sampled-action eligibility or self-action credit state.
        """
        if not self._teacher_credit_enabled:
            return {
                "applied": False,
                "refused": True,
                "reason": "teacher_credit_disabled",
                "credit_source": "teacher_demonstration",
                "eligibility_norm_before_credit": 0.0,
                "rewarded_update_norm": 0.0,
            }
        if self._teacher_credit_consumed:
            return {
                "applied": False,
                "refused": True,
                "reason": "teacher_credit_already_consumed",
                "credit_source": "teacher_demonstration",
                "eligibility_norm_before_credit": 0.0,
                "rewarded_update_norm": 0.0,
            }
        if not self._teacher_credit_pending:
            return {
                "applied": False,
                "refused": True,
                "reason": "teacher_credit_not_pending",
                "credit_source": "teacher_demonstration",
                "eligibility_norm_before_credit": 0.0,
                "rewarded_update_norm": 0.0,
            }
        if self._teacher_demo_channel is None or self._teacher_eligibility is None:
            return {
                "applied": False,
                "refused": True,
                "reason": "teacher_demonstration_not_armed",
                "credit_source": "teacher_demonstration",
                "eligibility_norm_before_credit": 0.0,
                "rewarded_update_norm": 0.0,
            }
        if not hasattr(self.plasticity_rule, "actor_delta") or self.plasticity_rule.name() not in (
            "reward_eprop_rate_adaptation",
            "inherited_learning_signal_generator",
        ):
            return {
                "applied": False,
                "refused": True,
                "reason": "teacher_credit_requires_eprop_family",
                "credit_source": "teacher_demonstration",
                "eligibility_norm_before_credit": 0.0,
                "rewarded_update_norm": 0.0,
            }

        demo_channel = int(self._teacher_demo_channel)
        elig = self._teacher_eligibility
        elig_norm = float(elig.norm().item())
        # Snapshot self eligibility so teacher credit cannot rewrite it.
        self_elig_before = self.eligibility.trace.detach().clone()
        w_before = self.action_ctx.W_motor.weight.data.clone()
        pre_stats = self._rewarded_action_stats(demo_channel)

        with torch.no_grad():
            _, motor_logits = self.action_ctx(self.rho.relational_repr, self.rho.action_repr)
            teaching = float(self._teacher_teaching_signal)
            delta_t = self.plasticity_rule.td_step(
                teaching,
                self.rho.relational_repr,
                is_terminal=True,
            )
            self.plasticity_rule.update_critic(delta_t, self.rho.relational_repr)
            actor_kwargs = dict(
                eligibility=elig,
                delta_t=delta_t,
                chosen_channel=demo_channel,
                motor_logits=motor_logits,
                n_channels=self.genome.n_motor_channels,
            )
            if self.plasticity_rule.name() == "inherited_learning_signal_generator":
                actor_kwargs["relational_state"] = self.rho.relational_repr
            dW = self.plasticity_rule.actor_delta(**actor_kwargs)
            if hasattr(self, "_r2_plasticity_channel_mask"):
                dW = dW * self._r2_plasticity_channel_mask
            if hasattr(self, "_r2_plasticity_mask_gain"):
                dW = dW * float(self._r2_plasticity_mask_gain)
            self.action_ctx.W_motor.weight.data.add_(dW)
            self._last_actor_delta = dW.detach().clone()
            if self.plasticity_rule.name() == "inherited_learning_signal_generator":
                self._last_learning_signal_norm = float(
                    self.plasticity_rule.learning_signal_per_unit(
                        delta_t,
                        demo_channel,
                        motor_logits,
                        relational_state=self.rho.relational_repr,
                    ).norm().item()
                )
            else:
                self._last_learning_signal_norm = float(
                    self.plasticity_rule.learning_signal_per_unit(
                        delta_t, demo_channel, motor_logits
                    ).norm().item()
                )
            self._last_td_error = float(delta_t.item())
            self._last_critic_value = float(
                self.plasticity_rule.critic.value(self.rho.relational_repr).item()
            )

        # Restore sampled-action eligibility unchanged.
        self.eligibility.trace.copy_(self_elig_before)

        update_norm = float(dW.norm().item())
        post_stats = self._rewarded_action_stats(demo_channel)
        margin_delta = post_stats["margin"] - pre_stats["margin"]
        outcome_sign = int((teaching > 0) - (teaching < 0))

        self._teacher_credit_consumed = True
        self._teacher_credit_pending = False

        result = {
            "applied": True,
            "refused": False,
            "reason": "",
            "credit_source": "teacher_demonstration",
            "actor_target": demo_channel,
            "eligibility_norm_before_credit": elig_norm,
            "rewarded_update_norm": update_norm,
            "signed_reward_projection": teaching * update_norm,
            "pre_rewarded_action_probability": pre_stats["probability"],
            "post_rewarded_action_probability": post_stats["probability"],
            "pre_rewarded_action_margin": pre_stats["margin"],
            "post_rewarded_action_margin": post_stats["margin"],
            "margin_change_sign": int((margin_delta > 0) - (margin_delta < 0)),
            "learning_signal_norm": float(self._last_learning_signal_norm),
            "critic_value": float(self._last_critic_value),
            "td_error": float(self._last_td_error),
            "action_entropy": float(getattr(self, "_last_action_entropy", 0.0)),
            "outcome_sign": outcome_sign,
            "teaching_signal": teaching,
            "self_eligibility_unchanged": bool(
                torch.allclose(self.eligibility.trace, self_elig_before)
            ),
        }
        self._last_teacher_credit = dict(result)
        return result

    def _rewarded_action_stats(self, rewarded_channel: int) -> dict[str, float]:
        """Read-only policy stats for the rewarded (chosen) motor channel."""
        with torch.no_grad():
            _, motor_logits = self.action_ctx(self.rho.relational_repr, self.rho.action_repr)
            temp = 1.0
            if hasattr(self.plasticity_rule, "temperature"):
                temp = float(self.plasticity_rule.temperature)
            import torch.nn.functional as F
            probs = F.softmax(motor_logits / max(temp, 1e-6), dim=-1)
            p = float(probs[rewarded_channel].item())
            top2 = probs.topk(min(2, probs.numel())).values
            margin = float((top2[0] - top2[-1]).item())
            # Channel-specific margin vs best alternative
            others = probs.clone()
            others[rewarded_channel] = -1.0
            alt = float(others.max().item())
            channel_margin = p - alt
            return {"probability": p, "margin": channel_margin, "top_margin": margin}

    def rest(self, n_ticks: int = 1) -> dict:
        """
        Internal rest period: replay and optional legacy plasticity.

        If outcome credit was already consumed via apply_outcome_credit(),
        rest() skips duplicate plasticity automatically.
        """
        if not self._outcome_credit_consumed:
            self._apply_local_plasticity()

        self._outcome_credit_consumed = False
        self._outcome_credit_pending = False
        self._awaiting_consequence = False

        replayed = 0
        if not self.consolidation_disabled and self.hippocampus._store:
            onset_frac = self.genome.schedule.consolidation_onset_age_frac
            if self.age_frac >= onset_frac:
                consol = self.consolidation
                if hasattr(consol, "replay"):
                    episodes = consol.replay.sample_episodes(
                        self.hippocampus._store, self._surprises
                    )
                elif hasattr(consol, "sample_episodes"):
                    episodes = consol.sample_episodes(
                        self.hippocampus._store, self._surprises
                    )
                else:
                    episodes = []
                replayed = len(episodes)
                self.slog.append(EventKind.REPLAY, step=self.step, payload={"n_replayed": replayed})
        return {"replayed": replayed, "step": self.step}

    def episode_reset(self) -> EpisodeReset:
        """
        Clear ρ and transient eligibility only.
        W and H persist unchanged. S_log is not cleared.

        Reset before outcome credit invalidates any pending credit token so
        stale eligibility cannot be used after a boundary reset.
        """
        if self._outcome_credit_pending and not self._outcome_credit_consumed:
            self._outcome_credit_pending = False
            self._awaiting_consequence = False
        self._teacher_awaiting_consequence = False
        self._teacher_credit_pending = False
        self._teacher_eligibility = None
        self._teacher_demo_channel = None
        snap = self.rho.cleared_snapshot()
        elig_snap = {"trace": self.eligibility.trace.cpu().tolist()}
        self.rho.reset()
        self.eligibility.reset_transient()
        if hasattr(self.plasticity_rule, "reset_episode"):
            self.plasticity_rule.reset_episode()
        self.hippocampus.reset_episode_counter()
        reset = EpisodeReset(
            cleared_working_state=snap["working_state"],
            cleared_transient_eligibility=elig_snap,
            step_at_reset=self.step,
        )
        self.slog.append(EventKind.RESET, step=self.step)
        return reset

    def full_checkpoint(self) -> FullCheckpoint:
        """
        Capture complete organism state.
        Used for exact restoration and matched-donor-twin construction.
        """
        cp = FullCheckpoint(
            genome_state=self.genome.to_dict(),
            cortex_state={
                "sensory": {k: v.cpu() for k, v in self.sensory_ctx.state_dict().items()},
                "relational": {k: v.cpu() for k, v in self.relational_ctx.state_dict().items()},
                "action": {k: v.cpu() for k, v in self.action_ctx.state_dict().items()},
            },
            hippocampus_state=self.hippocampus.hippocampus_state_dict(),
            working_state=self.rho.state_dict(),
            eligibility_state={"trace": self.eligibility.trace.cpu()},
            plasticity_state=self.hippocampus.hippocampus_plasticity_state_dict(),
            counters={
                "step": self.step,
                "outcome_credit_pending": self._outcome_credit_pending,
                "outcome_credit_consumed": self._outcome_credit_consumed,
                "awaiting_consequence": self._awaiting_consequence,
                "credit_interaction_step": self._credit_interaction_step,
                "last_action_channel": getattr(self, "_last_action_channel", -1),
            },
            rng_state={"torch": torch.get_rng_state().tolist()},
            slog_snapshot=self.slog.snapshot(),
        )
        self.slog.append(EventKind.CHECKPOINT, step=self.step)
        return cp

    def restore_from_checkpoint(self, cp: FullCheckpoint) -> None:
        """Restore organism to a previously saved full checkpoint."""
        self.sensory_ctx.load_state_dict(cp.cortex_state["sensory"])
        self.relational_ctx.load_state_dict(cp.cortex_state["relational"])
        self.action_ctx.load_state_dict(cp.cortex_state["action"])
        self.hippocampus.load_hippocampus_state_dict(cp.hippocampus_state)
        self.hippocampus.load_hippocampus_plasticity_state_dict(cp.plasticity_state)
        self.rho.load_state_dict(cp.working_state)
        self.eligibility.trace = cp.eligibility_state["trace"].to(self.device)
        self.step = cp.counters["step"]
        self.slog.restore_from_snapshot(cp.slog_snapshot)
        # Deterministic checkpoint rule: pending outcome credit is invalidated on restore.
        # Caller must re-deliver consequence (observe with reward) before credit applies.
        self._outcome_credit_pending = False
        self._outcome_credit_consumed = True
        self._awaiting_consequence = False
        self._credit_interaction_step = cp.counters.get("credit_interaction_step", -1)
        if "last_action_channel" in cp.counters and cp.counters["last_action_channel"] >= 0:
            self._last_action_channel = int(cp.counters["last_action_channel"])

    def hippocampal_graft(self, graft: HippocampalGraft) -> None:
        """
        Transfer only H and H-local plasticity state from a matched donor twin.
        Does NOT transfer: cortex W, genome G, S_log, rho, runner metadata.
        """
        self.hippocampus.load_hippocampus_state_dict(graft.donor_hippocampus_state)
        self.hippocampus.load_hippocampus_plasticity_state_dict(graft.donor_hippocampus_plasticity_state)
        self.slog.append(EventKind.GRAFT, step=self.step, payload={
            "donor_checkpoint_hash": graft.donor_checkpoint_hash,
        })

    def telemetry(self) -> OrganismTelemetry:
        """
        Read-only diagnostics snapshot.
        This object must never be fed back into observe(), reward, retrieval,
        or any teaching path. Audited dynamically in test_boundaries.py.
        """
        cap = self.hippocampus.capacity_telemetry()
        return OrganismTelemetry(
            activations={
                "sensory": self.rho.sensory_repr.detach().cpu().numpy(),
                "relational": self.rho.relational_repr.detach().cpu().numpy(),
                "action": self.rho.action_repr.detach().cpu().numpy(),
            },
            eligibility_state={
                "trace_norm": float(self.eligibility.trace.norm().item()),
                "last_actor_delta_norm": float(getattr(self, "_last_actor_delta", torch.zeros(1)).norm().item()),
            },
            replay_events=[],
            plasticity_events=[{
                "family": self.plasticity_rule.name(),
                "last_action_channel": getattr(self, "_last_action_channel", None),
            }],
            hippocampus_capacity_used=cap["capacity_used"],
            hippocampus_capacity_max=cap["capacity_max"],
            evictions_this_episode=cap["evictions_total"],
        )

    def _arm_teacher_demonstration(self, channel: int) -> None:
        """
        Efference-only demo injection plus separate teacher eligibility arming.

        Does not mutate sampled-action eligibility: live eligibility is restored
        after computing the teacher-owned eligibility snapshot.
        """
        n = self.genome.n_motor_channels
        if channel < 0 or channel >= n:
            return
        self._inject_observed_motor_event(channel)
        self_elig_backup = self.eligibility.trace.detach().clone()
        self.eligibility.update(self.rho.relational_repr, self.rho.action_repr)
        self._teacher_eligibility = self.eligibility.trace.detach().clone()
        self.eligibility.trace.copy_(self_elig_backup)
        self._teacher_demo_channel = int(channel)
        self._teacher_awaiting_consequence = True
        self._teacher_credit_pending = False
        self._teacher_credit_consumed = False

    def _inject_observed_motor_event(self, channel: int) -> None:
        """Efference-only teacher demonstration on the action pathway."""
        n = self.genome.n_motor_channels
        if channel < 0 or channel >= n:
            return
        demo = torch.zeros(n, device=self.device)
        demo[channel] = 1.0
        efference = demo @ self.action_ctx.W_motor.weight
        self.rho.action_repr = self.rho.action_repr + 0.25 * efference

    def count_mutable_scalars(self) -> dict:
        """Compute the six capacity metrics required by scale.py."""
        cortical = sum(
            p.numel() for m in [self.sensory_ctx, self.relational_ctx, self.action_ctx, self.neuromod]
            for p in m.parameters()
        )
        fast_syn = (
            self.hippocampus.spec.dg_n_units ** 2   # W_hebb
            + self.hippocampus.spec.capacity * (
                self.hippocampus.spec.ec_dim + self.genome.relational_ctx.n_units
            )
        )
        return {
            "mutable_cortical_scalars": cortical,
            "fast_synaptic_memory_scalars": fast_syn,
            "total_mutable_neural_scalars": cortical + fast_syn,
            "recurrent_state_dim": self.genome.relational_ctx.n_units,
            "activations_per_tick": (
                self.genome.sensory_ctx.n_units
                + self.genome.relational_ctx.n_units
                + self.genome.action_ctx.n_units
            ),
            "checkpoint_bytes": -1,   # filled by scale.py after serialisation
        }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_tensor(v, device: torch.device) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v.float().to(device)
    return torch.tensor(v, dtype=torch.float32, device=device)


def _pad_or_trim(v: torch.Tensor, target: int) -> torch.Tensor:
    n = v.numel()
    if n == target:
        return v
    if n > target:
        return v[:target]
    return F.pad(v, (0, target - n))
