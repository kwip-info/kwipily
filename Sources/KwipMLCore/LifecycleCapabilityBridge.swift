import Foundation

/// Describes existing Train and History actions; never registers a canvas component.
public enum LifecycleCapabilityBridge {
    public static func declaration() -> CapabilityDescriptor {
        let entry = BundledLifecycleCapabilities.entry
        let weights = ArtifactContract(kind: .weights, backend: .pytorch, format: "kwip.trained-package", resume: .weightsOnly)
        let checkpoint = ArtifactContract(kind: .trainingCheckpoint, backend: .pytorch, format: "kwip.private-training-checkpoint.v1", resume: .weightsOnly)
        return CapabilityDescriptor(id: entry.id, packageID: "kwip.lifecycle", backend: .pytorch,
            ports: [
                PortContract("savedWeights", direction: .output, schema: .artifact(weights), phases: [.export]),
                PortContract("savedContinuation", direction: .output, schema: .artifact(checkpoint), phases: [.checkpoint]),
                PortContract("parentWeights", direction: .input, schema: .artifact(weights), phases: [.resume]),
                PortContract("parentContinuation", direction: .input, schema: .artifact(checkpoint), phases: [.resume])
            ], lifecycle: LifecycleContract(phases: [.checkpoint, .resume, .export, .reset], resume: .weightsOnly),
            devices: [.cpu, .mps], availability: .existingPOCAdapter, adapterID: entry.adapterID,
            learning: LearningContract(conceptIDs: ["training.saved-settings", "training.continuation"],
                lessonIDs: entry.learning?.lessonIDs ?? [], exampleIDs: entry.learning?.exampleIDs ?? []))
    }
}
