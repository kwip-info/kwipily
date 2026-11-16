import Foundation

public enum NormalizationLessonTask {
    public static let assessmentID = "assessment.training-standardization"
    public static let objectiveID = "data.training-normalization"
    public static let currentVersion = ContractVersion(1,2,0)
    public static let allowedOperators: Set<String> = Set(["sourceCSV","cast","split","standardScalerFit","transform"].map { "kwip.data." + $0 })
    public static let publicTraining = LearningTensor(shape:[3,2],values:[0,10,2,12,4,14])
    public static let publicValidation = LearningTensor(shape:[1,2],values:[12,22])
    public static let instructions = "Fit only training rows. For each feature, subtract the training mean and divide by the square root of population variance (sum of squared deviations divided by training count). Use scale 1 for a constant training feature. Apply the same fitted state to both training and validation; held-out values must not change the fitted statistics."
    public static func plan(runtimeVersion:String,seed:UInt64,assessmentVersion:ContractVersion = currentVersion,contentVersion:ContractVersion = ContractVersion()) throws -> LearningAssessmentPlan {
        guard contentVersion == ContractVersion() else { throw LearningContractError("Unsupported normalization content revision.") }
        if assessmentVersion == ContractVersion() { return LearningExamples.trainingOnlyStandardization(runtimeVersion:runtimeVersion,seed:seed) }
        if assessmentVersion == ContractVersion(1,1,0) { return LearningExamples.trainingOnlyStandardizationV11(runtimeVersion:runtimeVersion,seed:seed) }
        guard assessmentVersion == currentVersion else { throw LearningContractError("Unsupported normalization assessment revision.") }
        return LearningExamples.trainingOnlyStandardizationV12(runtimeVersion:runtimeVersion,seed:seed)
    }
}

public struct NormalizationLearningSubmission: Codable, Equatable {
    public var schemaVersion = 1
    public var kind = "normalizationGraph"
    public var graph: DataFlowDocument
    public var outputs: [String:DataFlowEndpoint]
    public init(graph:DataFlowDocument,outputs:[String:DataFlowEndpoint]) { self.graph=graph;self.outputs=outputs }
    private enum CodingKeys:String,CodingKey { case schemaVersion,kind,graph,outputs }
    public init(from decoder:Decoder) throws {
        let raw=try decoder.singleValueContainer().decode(JSONValue.self)
        guard case .object(let object)=raw,Set(object.keys)==["schemaVersion","kind","graph","outputs"],case .integer(1)=object["schemaVersion"],let rawGraph=object["graph"],case .object(let outputs)=object["outputs"] else { throw LearningContractError("Invalid normalization submission fields.") }
        for value in outputs.values { guard case .object(let endpoint)=value,Set(endpoint.keys)==["node","port"] else { throw LearningContractError("Invalid normalization output fields.") } }
        let values=try decoder.container(keyedBy:CodingKeys.self)
        schemaVersion=try values.decode(Int.self,forKey:.schemaVersion);kind=try values.decode(String.self,forKey:.kind)
        graph=try DataFlowCodec.decodeGraph(JSONEncoder().encode(rawGraph))
        self.outputs=try values.decode([String:DataFlowEndpoint].self,forKey:.outputs)
        try validateDraft()
    }
    public func validateDraft() throws {
        try DataFlowCodec.validate(graph)
        guard schemaVersion == 1, kind == "normalizationGraph",
              Set(outputs.keys).isSubset(of:["state","training","validation"]),
              outputs.values.allSatisfy({ !$0.node.isEmpty && $0.node.count <= 128 && !$0.port.isEmpty && $0.port.count <= 128 }),
              graph.nodes.allSatisfy({ NormalizationLessonTask.allowedOperators.contains($0.operatorID) }) else { throw LearningContractError("Invalid bounded normalization draft graph or output references.") }
        for source in graph.nodes where source.operatorID == "kwip.data.sourceCSV" {
            guard Set(source.config.keys).isSubset(of:["separator"]), source.config["separator"] == nil || source.config["separator"] == .string(",") else { throw LearningContractError("Lesson source is an adapter-owned comma CSV; paths and external bindings are not allowed.") }
        }
    }
    public func validate() throws {
        try validateDraft()
        guard schemaVersion == 1, kind == "normalizationGraph", !graph.nodes.isEmpty,
              Set(outputs.keys) == ["state","training","validation"],
              graph.nodes.allSatisfy({ NormalizationLessonTask.allowedOperators.contains($0.operatorID) }),
              graph.nodes.filter({ $0.operatorID == "kwip.data.sourceCSV" }).count == 1 else {
            throw LearningContractError("Normalization requires one fixture source, allowed data operators and explicit state/training/validation outputs.")
        }
        let ports: [String:Set<String>] = ["sourceCSV":["table"],"cast":["table"],"split":["training","validation"],"standardScalerFit":["state"],"transform":["table"]]
        for (key, endpoint) in outputs {
            guard let node=graph.nodes.first(where:{$0.id == endpoint.node}),
                  ports[String(node.operatorID.dropFirst("kwip.data.".count))]?.contains(endpoint.port) == true,
                  (key == "state") == (endpoint.port == "state") else { throw LearningContractError("Selected normalization output does not exist or has the wrong type.") }
        }
    }
    public func semanticDigest() throws -> String {
        var canonical=self
        canonical.graph.nodes.sort { $0.id < $1.id }
        canonical.graph.edges.sort { $0.id < $1.id }
        return learningHash(try NormalizationLearningCodec.encodeSubmission(canonical))
    }
}

public struct NormalizationLearningDraft: Codable, Equatable {
    public var schemaVersion = 1
    public var assessmentID = NormalizationLessonTask.assessmentID
    public var contentVersion = ContractVersion()
    public var assessmentVersion = NormalizationLessonTask.currentVersion
    public var projectID:String
    public var returnProjectID:String?
    public var submission:NormalizationLearningSubmission
    public var mode:LearningMode
    public var notes = ""
    public var hintIDs:[String] = []
    public init(projectID:String,submission:NormalizationLearningSubmission,mode:LearningMode = .guided,returnProjectID:String? = nil) { self.projectID=projectID;self.submission=submission;self.mode=mode;self.returnProjectID=returnProjectID }
    public func validate() throws {
        guard schemaVersion == 1, assessmentID == NormalizationLessonTask.assessmentID, UUID(uuidString:projectID) != nil, returnProjectID.map({ UUID(uuidString:$0) != nil && $0 != projectID }) ?? true, notes.count <= 8000,
              hintIDs.count <= 16, Set(hintIDs).count == hintIDs.count, hintIDs.allSatisfy({ !$0.isEmpty && $0.count <= 128 }) else { throw LearningContractError("Invalid or unbounded normalization draft.") }
        _ = try NormalizationLessonTask.plan(runtimeVersion:"validation",seed:0,assessmentVersion:assessmentVersion,contentVersion:contentVersion)
        try submission.validateDraft()
    }
    public func executableSubmission() throws -> NormalizationLearningSubmission { try validate();try submission.validate();return submission }
}

public enum NormalizationLearningCodec {
    private static func encode<T:Encodable>(_ value:T) throws -> Data {
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        let data=try encoder.encode(value)
        guard data.count <= 2_000_000 else { throw LearningContractError("Normalization document exceeds 2 MB.") }
        return data
    }
    public static func encodeSubmission(_ value:NormalizationLearningSubmission) throws -> Data { try value.validate();return try encode(value) }
    public static func decodeSubmission(_ data:Data) throws -> NormalizationLearningSubmission { try decodeSubmission(data, executable:true) }
    private static func decodeSubmission(_ data:Data, executable:Bool) throws -> NormalizationLearningSubmission {
        guard data.count <= 2_000_000, case .object(let root)=try JSONDecoder().decode(JSONValue.self,from:data), Set(root.keys)==["schemaVersion","kind","graph","outputs"],case .integer(1)=root["schemaVersion"],case .object(let outputs)=root["outputs"] else { throw LearningContractError("Invalid normalization submission envelope.") }
        for endpoint in outputs.values { guard case .object(let object)=endpoint,Set(object.keys)==["node","port"] else { throw LearningContractError("Invalid normalization output reference.") } }
        guard let graph=root["graph"] else { throw LearningContractError("Missing normalization graph.") }
        _ = try DataFlowCodec.decodeGraph(JSONEncoder().encode(graph))
        let value=try JSONDecoder().decode(NormalizationLearningSubmission.self,from:data);if executable { try value.validate() } else { try value.validateDraft() };return value
    }
    public static func encodeReceipt(_ value:NormalizationLearningReceipt) throws -> Data {
        _ = try value.verifiedReplay(expectedProjectID:value.projectID)
        return try encode(value)
    }
    public static func decodeReceipt(_ data:Data,expectedProjectID:String) throws -> NormalizationLearningReceipt {
        guard data.count <= 2_000_000, case .object(let root)=try JSONDecoder().decode(JSONValue.self,from:data),
              Set(root.keys)==["schemaVersion","attempt","projectID","submission","submissionSHA256","runtime","fixtureSeed","fixtureDigest","evidence"],case .integer(1)=root["schemaVersion"] else { throw LearningContractError("Invalid normalization receipt envelope.") }
        let value=try JSONDecoder().decode(NormalizationLearningReceipt.self,from:data)
        _ = try value.verifiedReplay(expectedProjectID:expectedProjectID)
        return value
    }
    public static func encodeDraft(_ value:NormalizationLearningDraft) throws -> Data { try value.validate();return try encode(value) }
    public static func decodeDraft(_ data:Data) throws -> NormalizationLearningDraft {
        guard data.count <= 2_000_000 else { throw LearningContractError("Normalization document exceeds 2 MB.") }
        guard case .object(let object)=try JSONDecoder().decode(JSONValue.self,from:data),
              Set(object.keys).subtracting(["returnProjectID"]) == ["schemaVersion","assessmentID","contentVersion","assessmentVersion","projectID","submission","mode","notes","hintIDs"],
              let raw=object["submission"] else { throw LearningContractError("Invalid normalization draft fields.") }
        _ = try decodeSubmission(JSONEncoder().encode(raw),executable:false)
        let value=try JSONDecoder().decode(NormalizationLearningDraft.self,from:data);try value.validate();return value
    }
}

public struct NormalizationLearningReceipt: Codable {
    public var schemaVersion = 1
    public var attempt:LearningAttempt
    public var projectID:String
    public var submission:NormalizationLearningSubmission
    public var submissionSHA256:String
    public var runtime:LearningRuntime
    public var fixtureSeed:UInt64
    public var fixtureDigest:String
    public var evidence:LearningEvidence
    public init(attempt:LearningAttempt,projectID:String,submission:NormalizationLearningSubmission,submissionSHA256:String,runtime:LearningRuntime,fixtureSeed:UInt64,fixtureDigest:String,evidence:LearningEvidence) {
        self.attempt=attempt;self.projectID=projectID;self.submission=submission;self.submissionSHA256=submissionSHA256;self.runtime=runtime;self.fixtureSeed=fixtureSeed;self.fixtureDigest=fixtureDigest;self.evidence=evidence
    }
    public func verifiedReplay(expectedProjectID:String) throws -> LearningAssessmentPlan {
        guard schemaVersion == 1, UUID(uuidString:projectID) != nil, projectID == expectedProjectID,
              UUID(uuidString:attempt.id) != nil, attempt.timestamp.timeIntervalSince1970.isFinite,
              attempt.hintIDs.count <= 16, Set(attempt.hintIDs).count == attempt.hintIDs.count,
              attempt.hintIDs.allSatisfy({ !$0.isEmpty && $0.count <= 128 }),
              try learningHash(NormalizationLearningCodec.encodeSubmission(submission)) == submissionSHA256 else {
            throw LearningContractError("Normalization receipt project, attempt or submission identity differs.")
        }
        let plan=try NormalizationLessonTask.plan(runtimeVersion:runtime.version,seed:fixtureSeed,assessmentVersion:attempt.assessmentVersion,contentVersion:attempt.contentVersion)
        guard attempt.objectiveID == plan.objectiveID, runtime == plan.runtime, fixtureDigest == plan.fixtureDigest,
              evidence.assessmentID == plan.id, evidence.assessmentVersion == plan.assessmentVersion,
              evidence.runtime == runtime, evidence.fixtureDigest == fixtureDigest, evidence.submissionDigest == submissionSHA256,
              try LearningAssessor.evaluate(plan,evidence:evidence) == attempt.grade else {
            throw LearningContractError("Normalization receipt task, runtime, fixtures or recomputed grade differ.")
        }
        return plan
    }
    public func formativeProjection(expectedProjectID:String,currentSubmission:NormalizationLearningSubmission,currentMode:LearningMode) -> NormalizationFormativeProjection? {
        guard currentMode != .assessment, attempt.mode != .assessment,
              let current=try? currentSubmission.semanticDigest(),let recorded=try? submission.semanticDigest(),current == recorded,
              let plan=try? verifiedReplay(expectedProjectID:expectedProjectID),
              attempt.grade.status == .passed || attempt.grade.status == .failed else { return nil }
        var projected:[NormalizationFormativeFixture]=[]
        for fixture in plan.fixtures {
            guard let measured=evidence.fixtures.first(where:{$0.fixtureID == fixture.id}),
                  Set(measured.tensors.keys) == Set(fixture.expected.keys),
                  measured.tensors.allSatisfy({ key,value in value.wellFormed && value.shape == fixture.expected[key]?.shape }),
                  let training=fixture.inputs["training"],let validation=fixture.inputs["validation"] else { return nil }
            projected.append(NormalizationFormativeFixture(id:fixture.id,training:training,validation:validation,measured:measured.tensors,expected:fixture.expected))
        }
        return NormalizationFormativeProjection(fixtures:projected,submissionDigest:submissionSHA256)
    }
}

/// View-only values, intentionally not Codable. Assessment receipts never project hidden fixtures.
public struct NormalizationFormativeFixture: Equatable, Identifiable {
    public let id:String
    public let training:LearningTensor
    public let validation:LearningTensor
    public let measured:[String:LearningTensor]
    public let expected:[String:LearningTensor]
}
public struct NormalizationFormativeProjection: Equatable {
    public let fixtures:[NormalizationFormativeFixture]
    public let submissionDigest:String
}
