import Foundation

public enum ModuleCompositionLearningTask: String, CaseIterable, Codable {
    case buildReuse = "composition-build-reuse"
    case orderName = "composition-order-name"
    public var title: String { self == .buildReuse ? "Build and reuse a block" : "Order and name the work" }
    public var instructions: String { self == .buildReuse ? "Build a reusable block, then connect two private copies. Learned settings are weights and offsets adjusted during training. A private copy has its own settings. Influence means how a small change affects a result. Compare results, influence, adjustments, and saved reuse. The supplied sign-reversal and subtraction layers are fixed. A supplied behavior check runs alongside the main calculation; it checks training versus evaluation without changing your main result." : "Connect A then B. Learned settings are weights and offsets adjusted during training. Influence means how a small change affects a result. A sequence follows its connected order. An ordered or named collection stores children; its connections say what work to do. Compare storage and results separately." }
    public var measuredShapes: [String: [Int]] {
        self == .buildReuse ? ["branchOutputs":[3,3],"nestedInfluences":[11],"privateAdjustment":[8],"modePropagation":[3,2],"savedReuse":[9]] : ["sequentialInfluences":[10],"orderedOutputs":[3,3],"namedOutputs":[3,2],"structuralIdentity":[1],"savedReuse":[3,3]]
    }
    public func assessment(seed: UInt64, runtimeVersion: String) throws -> LearningAssessmentPlan {
        let base:[[Double]] = [[2,1,3,-2],[-1,0.5,2,1],[0.5,-2,-2,0.25],[3,-0.5,0.25,2]]
        let rows:[[Double]] = [[-2,0.5,3],[-1.5,0.25,2],[-3,1,2.5],[-0.75,0.5,1.75]]
        let upstream:[[Double]] = [[1,-2,0.5],[-0.5,1,2],[2,-0.25,-1],[-1,0.5,1.5]]
        var state = seed
        func draw() -> Double { state = state &* 6364136223846793005 &+ 1442695040888963407; return Double(Int((state >> 32) % 9)-4)/32 }
        let fixtures = try (0..<4).map { index -> LearningFixture in
            var p = base[index].map { $0 + draw() }
            if p[2]*p[1]+p[3] == p[0]*p[3]+p[1] { p[3] += 0.125 }
            let x = rows[index].enumerated().map { Double($0.offset)/1024 + $0.element + draw() }
            let u = upstream[index].map { $0 + draw() }
            _ = try ModuleCompositionLessonReference.output(coefficients:p,rows:x,kind:"branch")
            let inputs:[String:LearningTensor] = ["coefficients":.init(shape:[4],values:p),"rows":.init(shape:[3,1],values:x),"upstream":.init(shape:[3,1],values:u),"targets":.init(shape:[3,1],values:[0.5,-0.25,1])]
            let expected = measuredShapes.mapValues { LearningTensor(shape:$0,values:Array(repeating:0,count:$0.reduce(1,*))) }
            return .init(id:"\(rawValue)-\(seed)-\(index)",inputs:inputs,expected:expected)
        }
        let criteria = measuredShapes.keys.sorted().map { key in LearningCriterion(id:key,description:Self.label(key),nextStep:instructions,rule:.tensor(key:key,absoluteTolerance:key == "structuralIdentity" ? 0:2e-5,relativeTolerance:0)) }
        let plan = LearningAssessmentPlan(id:"assessment."+rawValue,objectiveID:"composition."+rawValue,task:instructions,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:["moduleComposition":ContractVersion()]),budget:.init(milliseconds:60_000,maximumTensorElements:4096),fixtureSeed:seed,fixtures:fixtures,criteria:criteria)
        try plan.validate(); return plan
    }
    public static func label(_ key:String)->String {
        switch key {
        case "branchOutputs":return "Follow both branches and named results"
        case "nestedInfluences":return "Measure each input and private setting's influence"
        case "privateAdjustment":return "Adjust the left copy and preserve the right copy"
        case "modePropagation":return "Switch layer behavior while preserving fixed settings"
        case "sequentialInfluences":return "Run the sequence in its connected order"
        case "orderedOutputs":return "Separate stored order from connected work"
        case "namedOutputs":return "Use real child names before and after renaming"
        case "structuralIdentity":return "Keep exact child ownership, names and order"
        case "savedReuse":return "Reuse genuine saved settings and standalone architecture"
        default:return key
        }
    }
}

public enum ModuleCompositionLessonReference {
    public static func output(coefficients: [Double], rows: [Double], kind: String) throws -> [Double] {
        guard coefficients.count == 4, !rows.isEmpty, rows.count <= 32, (coefficients+rows).allSatisfy({$0.isFinite && abs($0)<=100}),["branch","sequenceAB","sequenceBA"].contains(kind) else { throw LearningContractError("Use complete finite supplied affine settings and observation rows.") }
        let a=coefficients[0],b=coefficients[1],c=coefficients[2],d=coefficients[3]
        return rows.map { x in kind == "branch" ? (c+1)*(a*x+b)+d : kind == "sequenceAB" ? c*(a*x+b)+d : a*(c*x+d)+b }
    }
}

/// Current difference-table admission only; actual runtime receipts are verified separately.
public struct ModuleCompositionFormativeProjection: Equatable {
    public let submissionDigest: String
    public let fixtures: [LearningFixtureEvidence]
    public static func project(plan:LearningAssessmentPlan,evidence:LearningEvidence,currentSubmission:Data,registry:ComponentRegistry,mode:LearningMode,attemptMode:LearningMode)->Self? {
        guard mode != .assessment,mode == attemptMode,
              let task=ModuleCompositionLearningTask.allCases.first(where:{plan.id == "assessment."+$0.rawValue}),
              let expected=try? task.assessment(seed:plan.fixtureSeed,runtimeVersion:plan.runtime.version),expected == plan,
              (try? ModuleCompositionLearningSubmission.decode(currentSubmission,task:task,registry:registry)) != nil,
              evidence.assessmentID == plan.id,evidence.assessmentVersion == plan.assessmentVersion,
              evidence.fixtureDigest == plan.fixtureDigest,evidence.runtime == plan.runtime,
              evidence.submissionDigest == learningHash(currentSubmission),!evidence.cancelled,evidence.backendError == nil,
              evidence.elapsedMilliseconds.isFinite,(0...plan.budget.milliseconds).contains(evidence.elapsedMilliseconds),
              (0...plan.budget.maximumTensorElements).contains(evidence.maximumTensorElements),
              evidence.fixtures.count == plan.fixtures.count,Set(evidence.fixtures.map(\.fixtureID)).count == evidence.fixtures.count else {return nil}
        for fixture in plan.fixtures {
            guard let actual=evidence.fixtures.first(where:{$0.fixtureID == fixture.id}),actual.partitions.isEmpty,
                  Set(actual.tensors.keys) == Set(fixture.expected.keys),actual.tensors.allSatisfy({$0.value.wellFormed && $0.value.shape == fixture.expected[$0.key]?.shape}) else {return nil}
        }
        return .init(submissionDigest:evidence.submissionDigest,fixtures:plan.fixtures.compactMap { f in evidence.fixtures.first {$0.fixtureID == f.id} })
    }
}
