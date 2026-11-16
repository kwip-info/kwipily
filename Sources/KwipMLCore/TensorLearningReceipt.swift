import Foundation

/// Local replay evidence, not a signature or proof against a malicious local author.
/// Expected fixtures are regenerated from the recorded task version and seed.
public struct TensorLearningReceipt:Codable {
    public var schemaVersion=1
    public var projectID:String
    public var operation:String
    public var attempt:LearningAttempt
    public var submissionData:Data
    public var submissionSHA256:String
    public var runtime:LearningRuntime
    public var fixtureSeed:UInt64
    public var fixtureDigest:String
    public var evidence:LearningEvidence
    public var compositionRegistry:ComponentRegistry?
    public var compositionRegistrySHA256:String?

    public init(projectID:String,operation:String,mode:LearningMode,submissionData:Data,
                plan:LearningAssessmentPlan,evidence:LearningEvidence,compositionRegistry:ComponentRegistry?=nil)throws {
        self.compositionRegistry=compositionRegistry
        self.compositionRegistrySHA256=try compositionRegistry.map {learningHash(try Self.registryData($0))}
        self.projectID=projectID;self.operation=operation;self.submissionData=submissionData
        submissionSHA256=learningHash(submissionData);runtime=plan.runtime
        fixtureSeed=plan.fixtureSeed;fixtureDigest=plan.fixtureDigest;self.evidence=evidence
        attempt=LearningAttempt(objectiveID:plan.objectiveID,contentVersion:plan.contentVersion,
            assessmentVersion:plan.assessmentVersion,mode:mode,grade:try LearningAssessor.evaluate(plan,evidence:evidence))
        _ = try verifiedReplay(expectedProjectID:projectID)
    }
    @discardableResult public func verifiedReplay(expectedProjectID:String)throws->LearningAssessmentPlan {
        let compositionTask=ModuleCompositionLearningTask(rawValue:operation)
        if compositionTask != nil {
            guard let compositionRegistry,let compositionRegistrySHA256,learningHash(try Self.registryData(compositionRegistry))==compositionRegistrySHA256 else {throw LearningContractError("Composition replay needs its unchanged recorded registry snapshot and canonical hash.")}
        } else {
            guard compositionRegistry==nil,compositionRegistrySHA256==nil,evidence.fixtures.allSatisfy({$0.composition==nil}) else {throw LearningContractError("Composition registry and native metadata belong only to Block composition exercises.")}
        }
        let layerNormTask=LayerNormLearningTask(rawValue:operation)
        let embeddingTask=EmbeddingLearningTask(rawValue:operation)
        let matrixTask=MatrixLearningTask(rawValue:operation)
        func canonicalSubmission() throws -> Data {
            if let task=DataLoadingLearningTask(rawValue:operation) {return try DataLoadingLearningSubmission.decode(submissionData,task:task).encoded()}
            if let task=ProgramExportLearningTask(rawValue:operation) {return try ProgramExportLearningSubmission.decode(submissionData,task:task).encoded()}
            if let task=compositionTask,let compositionRegistry {return try ModuleCompositionLearningSubmission.decode(submissionData,task:task,registry:compositionRegistry).encoded()}
            if let task=GradientLearningTask(rawValue:operation) {return try GradientLearningSubmission.decode(submissionData,task:task).encoded()}
            if let task=InitializationLearningTask(rawValue:operation) {return try InitializationLearningSubmission.decode(submissionData,task:task).encoded()}
            if let task=CheckpointLearningTask(rawValue:operation) {return try CheckpointLearningSubmission.decode(submissionData,task:task).encoded()}
            if let task=LossLearningTask(rawValue:operation) {return try LossLearningSubmission.decode(submissionData,task:task).encoded()}
            if let task=AttentionLearningTask(rawValue:operation) {return try AttentionLearningSubmission.decode(submissionData,task:task).encoded()}
            if let task=RecurrentLearningTask(rawValue:operation) {return try RecurrentLearningSubmission.decode(submissionData,task:task).encoded()}
            if let task=SchedulerLearningTask(rawValue:operation) {return try SchedulerLearningSubmission.decode(submissionData,task:task).encoded()}
            if let task=OptimizerLearningTask(rawValue:operation) {return try OptimizerLearningSubmission.decode(submissionData,task:task).encoded()}
            if let task=StatefulNormalizationLearningTask(rawValue:operation) {return try StatefulNormalizationLearningSubmission.decode(submissionData,task:task).encoded()}
            if let task=SpatialPatchLearningTask(rawValue:operation) {return try SpatialPatchLearningSubmission.decode(submissionData,task:task).encoded()}
            if let task=TensorRoutingLearningTask(rawValue:operation) {return try TensorRoutingLearningSubmission.decode(submissionData,task:task).encoded()}
            if let task=RemainingActivationLearningTask(rawValue:operation) {return try RemainingActivationLearningSubmission.decode(submissionData,task:task).encoded()}
            if let task=TensorCompositionLearningTask(rawValue:operation) {return try TensorCompositionLearningSubmission.decode(submissionData,task:task).encoded()}
            if let task=AdaptivePoolingLearningTask(rawValue:operation) {return try AdaptivePoolingLearningSubmission.decode(submissionData,task:task).encoded()}
            if let task=ConfigurableActivationLearningTask(rawValue:operation) {return try ConfigurableActivationLearningSubmission.decode(submissionData,task:task).encoded()}
        if let task=PoolingLearningTask(rawValue:operation) {return try PoolingLearningSubmission.decode(submissionData,task:task).encoded()}
            if let task=ConvolutionLearningTask(rawValue:operation) {return try ConvolutionLearningSubmission.decode(submissionData,task:task).encoded()}
            if operation == LazyLinearAssessment.operation {return try LazyLinearLearningSubmission.decode(submissionData).encoded()}
            if BilinearLearningTask(rawValue:operation) != nil {return try BilinearLearningSubmission.decode(submissionData).encoded()}
            if matrixTask != nil {return try MatrixLearningSubmission.decode(submissionData).encoded()}
            if embeddingTask != nil {return try EmbeddingLearningSubmission.decode(submissionData).encoded()}
            if EmbeddingBagLearningTask(rawValue:operation) != nil {return try EmbeddingBagLearningSubmission.decode(submissionData).encoded()}
            if layerNormTask != nil {return try LayerNormLearningSubmission.decode(submissionData).encoded()}
            return try TensorLearningSubmission.decode(submissionData,allowSoftmax:TensorLearningTask.softmaxAxis(for:operation) != nil).encoded()
        }
        guard schemaVersion == 1,UUID(uuidString:projectID) != nil,projectID == expectedProjectID,
              UUID(uuidString:attempt.id) != nil,attempt.timestamp.timeIntervalSince1970.isFinite,
              attempt.hintIDs.isEmpty,attempt.contentVersion == ContractVersion(),attempt.assessmentVersion == ContractVersion(),
              submissionData.count<=2_000_000,learningHash(submissionData) == submissionSHA256,
              try canonicalSubmission() == submissionData else {
            throw LearningContractError("Tensor receipt identity, version or canonical submission differs.")
        }
        let plan:LearningAssessmentPlan
        if let matrixTask {plan=try MatrixAssessment.plan(task:matrixTask,runtimeVersion:runtime.version,seed:fixtureSeed)}
        else if let embeddingTask {plan=try EmbeddingAssessment.plan(task:embeddingTask,runtimeVersion:runtime.version,seed:fixtureSeed)}
        else if let layerNormTask {plan=try LayerNormAssessment.plan(task:layerNormTask,runtimeVersion:runtime.version,seed:fixtureSeed)}
        else {plan=try TensorLearningTask.plan(operation:operation,runtimeVersion:runtime.version,seed:fixtureSeed)}
        let checkpointTask=CheckpointLearningTask(rawValue:operation)
        let expectedKeys=checkpointTask.map {Set($0.measuredShapes.keys)} ?? Set(plan.fixtures[0].expected.keys)
        guard attempt.objectiveID == plan.objectiveID,runtime == plan.runtime,fixtureDigest == plan.fixtureDigest,
              evidence.assessmentID == plan.id,evidence.assessmentVersion == plan.assessmentVersion,
              evidence.runtime == runtime,evidence.fixtureDigest == fixtureDigest,evidence.submissionDigest == submissionSHA256,
              evidence.fixtures.count == plan.fixtures.count,evidence.fixtures.allSatisfy({
                  $0.partitions.isEmpty && Set($0.tensors.keys) == expectedKeys &&
                  $0.tensors.values.allSatisfy {$0.values.count<=4096} &&
                  (checkpointTask == nil || $0.tensors.allSatisfy { key,value in value.wellFormed && value.shape == checkpointTask!.measuredShapes[key] })
              }),
              [.passed,.failed].contains(attempt.grade.status),
              try LearningAssessor.evaluate(plan,evidence:evidence) == attempt.grade else {
            throw LearningContractError("Tensor receipt task, measured evidence or recomputed grade differs.")
        }
        if let task=compositionTask,let compositionRegistry {
            let source=try ModuleCompositionLearningSubmission.decode(submissionData,task:task,registry:compositionRegistry).source
            for fixture in evidence.fixtures {
                guard let metadata=fixture.composition else {throw LearningContractError("Composition replay requires the measured native structure for every fixture.")}
                try metadata.validate(source:source,task:task,registry:compositionRegistry)
            }
        }
        return plan
    }
    private static func registryData(_ registry:ComponentRegistry)throws->Data {
        guard registry.schemaVersion==2,(1...256).contains(registry.components.count),Set(registry.components.map(\.id)).count==registry.components.count,registry.components.allSatisfy({!$0.id.isEmpty && $0.id.utf8.count<=512 && $0.inputs.count<=128 && $0.outputs.count<=128 && $0.fields.count<=128}) else {throw LearningContractError("Composition replay requires a bounded, uniquely identified captured registry.")}
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        let bytes=try encoder.encode(registry)
        guard bytes.count<=2_000_000 else {throw LearningContractError("Recorded registry exceeds the existing receipt budget.")}
        return bytes
    }
    public func encoded()throws->Data {
        try verifiedReplay(expectedProjectID:projectID)
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        let data=try encoder.encode(self)
        guard data.count<=2_000_000 else {throw LearningContractError("Tensor receipt exceeds 2 MB.")}
        return data
    }
    public static func decode(_ data:Data)throws->Self {
        guard data.count<=2_000_000 else {throw LearningContractError("Tensor receipt exceeds 2 MB.")}
        guard case .object(let raw)=try JSONDecoder().decode(JSONValue.self,from:data),
              Set(raw.keys).subtracting(["compositionRegistry","compositionRegistrySHA256"])==["schemaVersion","projectID","operation","attempt","submissionData","submissionSHA256","runtime","fixtureSeed","fixtureDigest","evidence"] else {
            throw LearningContractError("Tensor receipt has missing or unknown envelope fields.")
        }
        let receipt=try JSONDecoder().decode(Self.self,from:data)
        if let rawRegistry=raw["compositionRegistry"] {
            guard let registry=receipt.compositionRegistry,try JSONDecoder().decode(JSONValue.self,from:Self.registryData(registry))==rawRegistry else {throw LearningContractError("Recorded registry contains unknown, null, or noncanonical fields.")}
        }
        if raw["compositionRegistrySHA256"] != nil,receipt.compositionRegistrySHA256==nil {throw LearningContractError("Recorded registry hash cannot be null.")}
        try receipt.verifiedReplay(expectedProjectID:receipt.projectID)
        return receipt
    }
}
