import Foundation

/// Source is authored structure; fixture settings and measured answers never enter this envelope.
public struct ModuleCompositionLearningSubmission: Codable, Equatable {
    public let schemaVersion: Int
    public let kind: String
    public let taskID: String
    public let source: CompositeProjectDocument
    public init(source: CompositeProjectDocument, task: ModuleCompositionLearningTask, registry: ComponentRegistry) throws {
        self.schemaVersion = 1; self.kind = "compositionLearningGraph"; self.taskID = task.rawValue
        self.source = try Self.validatedSource(source, task: task, registry: registry, complete: true)
    }
    public static func validatedSource(_ source: CompositeProjectDocument, task: ModuleCompositionLearningTask, registry: ComponentRegistry, complete: Bool) throws -> CompositeProjectDocument {
        try ModuleCompositionLearningDraft.validateSource(source,task:task)
        if complete { _ = try CompositeExpander.expand(source,registry:registry,context:.explicitObjective) }
        return source
    }
    public func encoded() throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }
    public static func decode(_ data: Data, task: ModuleCompositionLearningTask, registry: ComponentRegistry) throws -> Self {
        guard data.count<=2_000_000,let object=try JSONSerialization.jsonObject(with:data) as? [String:Any],Set(object.keys)==["schemaVersion","kind","taskID","source"] else {throw LearningContractError("Use only the current source-qualified lesson envelope.")}
        let decoded=try JSONDecoder().decode(Self.self,from:data)
        guard decoded.schemaVersion==1,decoded.kind=="compositionLearningGraph",decoded.taskID==task.rawValue else {throw LearningContractError("Use this lesson's current submission identity.")}
        // Reject unknown nested fields that Codable would otherwise discard.
        let canonical=try JSONSerialization.jsonObject(with:decoded.encoded())
        func sameKeys(_ actual:Any,_ normalized:Any)->Bool {
            if let a=actual as? [String:Any],let b=normalized as? [String:Any] { return Set(a.keys)==Set(b.keys) && a.allSatisfy { key,value in sameKeys(value,b[key]!) } }
            if let a=actual as? [Any],let b=normalized as? [Any] { return a.count==b.count && zip(a,b).allSatisfy {sameKeys($0.0,$0.1)} }
            return true
        }
        guard sameKeys(object,canonical) else {throw LearningContractError("Unknown source or answer fields cannot enter this lesson.")}
        return try Self(source:decoded.source,task:task,registry:registry)
    }
}
