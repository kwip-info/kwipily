import KwipMLCore

/// The original bundle metadata tests retain their accepted pre-extension cohort.
/// RMSNorm, Rotary position and MoE have their own authored/runtime contract tests.
func fixtureModelComponentsBeforeTextExtensions(_ registry:ComponentRegistry)->[ComponentDefinition] {
    registry.components.filter { !["rms_norm","rotary_position","mixture_experts"].contains($0.id) }
}
