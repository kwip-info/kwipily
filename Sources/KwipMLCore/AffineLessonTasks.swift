import Foundation

public struct ModelConstructionLessonIdentity: Codable, Equatable {
    public var assessmentID: String
    public var contentVersion: ContractVersion
    public var taskSeed: UInt64?
    public init(assessmentID: String, contentVersion: ContractVersion = ContractVersion(), taskSeed: UInt64? = nil) {
        self.assessmentID = assessmentID; self.contentVersion = contentVersion; self.taskSeed = taskSeed
    }
    public static let relu = Self(assessmentID:"assessment.activation-relu")
    public static let tanh = Self(assessmentID:"assessment.activation-tanh")
    public static let sigmoid = Self(assessmentID:"assessment.activation-sigmoid")
    public static let geluExact = Self(assessmentID:"assessment.activation-gelu-exact")
    public static let geluTanh = Self(assessmentID:"assessment.activation-gelu-tanh")
    public var geluApproximateTanh:Bool? {
        if self == .geluExact {return false}
        if self == .geluTanh {return true}
        return nil
    }
    public static let frozenStandardization = Self(assessmentID: "assessment.frozen-standardization")
    public static let backwardChain = Self(assessmentID: "assessment.affine-backward-chain")
    public static let composition = Self(assessmentID: "assessment.affine-composition")
    public static func recovery(taskSeed: UInt64) -> Self { Self(assessmentID: "assessment.affine-from-probes", taskSeed: taskSeed) }
    public func validate() throws {
        guard contentVersion == ContractVersion(),
              ([Self.frozenStandardization.assessmentID,Self.composition.assessmentID,Self.backwardChain.assessmentID,Self.geluExact.assessmentID,Self.geluTanh.assessmentID,Self.sigmoid.assessmentID,Self.tanh.assessmentID,Self.relu.assessmentID].contains(assessmentID) && taskSeed == nil) ||
              (assessmentID == "assessment.affine-from-probes" && taskSeed != nil) else {
            throw LearningContractError("Unknown model-construction lesson identity, content version or task seed.")
        }
    }
}
/// Source compatibility only; historical Codable fields and wire identities remain unchanged.
public typealias AffineLessonIdentity = ModelConstructionLessonIdentity
public typealias AffineLessonTask = ModelConstructionLessonTask
public typealias AffineLessonTasks = ModelConstructionLessonTasks

public struct AffineProbe: Equatable, Identifiable {
    public let id: String
    public let input: [Double]
    public let output: [Double]
}
public struct ModelConstructionLessonTask: Equatable {
    public let identity: AffineLessonIdentity
    public let title: String
    public let instructions: String
    public let probes: [AffineProbe]
    public let allowedNodeTypes: Set<String>
}

/// A fixed allowlist of reviewed tasks; never dispatches arbitrary executable lesson content.
public enum ModelConstructionLessonTasks {
    /// Reviewed operator/mode constraints, independent of a preferred graph topology.
    public static func validateConstruction(_ graph:GraphDocument,identity:ModelConstructionLessonIdentity)throws {
        let task=try publicTask(identity:identity)
        guard graph.nodes.allSatisfy({task.allowedNodeTypes.contains($0.type)}) else {throw LearningContractError("This lesson permits only its declared construction components; remove unsupported operators.")}
        if identity == .sigmoid {
            let activations = graph.nodes.filter { $0.type == "sigmoid" }
            guard !activations.isEmpty, activations.allSatisfy({ $0.config.isEmpty }) else {
                throw LearningContractError("This task requires Sigmoid with no configuration settings.")
            }
            var ancestry = Set(graph.nodes.filter { $0.type == "output" }.map(\.id))
            var pending = Array(ancestry)
            while let node = pending.popLast() {
                for edge in graph.edges where edge.target == node {
                    if ancestry.insert(edge.source).inserted { pending.append(edge.source) }
                }
            }
            guard activations.contains(where: { ancestry.contains($0.id) }) else {
                throw LearningContractError("Connect a Sigmoid on the path that reaches Output; an unused block does not perform the requested transformation.")
            }
        }
        if identity == .tanh {
            let activations = graph.nodes.filter { $0.type == "tanh" }
            guard !activations.isEmpty, activations.allSatisfy({ $0.config.isEmpty }) else {
                throw LearningContractError("This task requires Tanh with no configuration settings.")
            }
            var ancestry = Set(graph.nodes.filter { $0.type == "output" }.map(\.id))
            var pending = Array(ancestry)
            while let node = pending.popLast() {
                for edge in graph.edges where edge.target == node {
                    if ancestry.insert(edge.source).inserted { pending.append(edge.source) }
                }
            }
            guard activations.contains(where: { ancestry.contains($0.id) }) else {
                throw LearningContractError("Connect a Tanh on the path that reaches Output; an unused block does not perform the requested transformation.")
            }
        }
        if identity == .relu {
            let activations = graph.nodes.filter { $0.type == "relu" }
            guard !activations.isEmpty, activations.allSatisfy({ $0.config.isEmpty }) else {
                throw LearningContractError("This task requires ReLU with no configuration settings.")
            }
            var ancestry = Set(graph.nodes.filter { $0.type == "output" }.map(\.id))
            var pending = Array(ancestry)
            while let node = pending.popLast() {
                for edge in graph.edges where edge.target == node {
                    if ancestry.insert(edge.source).inserted { pending.append(edge.source) }
                }
            }
            guard activations.contains(where: { ancestry.contains($0.id) }) else {
                throw LearningContractError("Connect a ReLU on the path that reaches Output; an unused block does not perform the requested transformation.")
            }
        }
        if let mode=identity.geluApproximateTanh {
            let activations=graph.nodes.filter{$0.type == "gelu"}
            guard !activations.isEmpty,activations.allSatisfy({$0.config["approximate_tanh"] == .bool(mode)}) else {throw LearningContractError("This task requires at least one GELU, with every GELU using Tanh approximation \(mode ? "on":"off").")}
        }
    }
    private static func target(_ seed: UInt64) -> ([Double], [Double]) {
        var generator = ProbeGenerator(state: seed)
        let weight = (0..<4).map { _ in generator.coefficient() }
        var bias = (0..<2).map { _ in generator.coefficient() }
        if bias == [0, 0] { bias[0] = 1 }
        return (weight, bias)
    }
    private static func apply(_ x: [Double], target: ([Double], [Double])) -> [Double] {
        (0..<2).map { target.0[$0*2] * x[0] + target.0[$0*2+1] * x[1] + target.1[$0] }
    }
    public static func publicTask(identity: AffineLessonIdentity) throws -> AffineLessonTask {
        try identity.validate()
        if identity == .sigmoid {
            return ModelConstructionLessonTask(identity: identity, title: "Transform values with Sigmoid",
                instructions: LearningExamples.sigmoidReference.instructions, probes: [],
                allowedNodeTypes: ["input", "linear", "sigmoid", "add", "concat", "output"])
        }
        if identity == .tanh {
            return ModelConstructionLessonTask(identity: identity, title: "Transform values with Tanh",
                instructions: LearningExamples.tanhReference.instructions, probes: [],
                allowedNodeTypes: ["input", "linear", "tanh", "add", "concat", "output"])
        }
        if identity == .relu {
            return ModelConstructionLessonTask(identity: identity, title: "Transform values with ReLU",
                instructions: LearningExamples.reluReference.instructions, probes: [],
                allowedNodeTypes: ["input", "linear", "relu", "add", "concat", "output"])
        }
        if identity == .composition {
            return AffineLessonTask(identity: identity, title: "Compose affine maps", instructions: LearningExamples.affineReference.instructions,
                                    probes: [], allowedNodeTypes: ["input", "linear", "add", "concat", "relu", "tanh", "output"])
        }
        if identity == .frozenStandardization {
            return ModelConstructionLessonTask(identity: identity, title: "Carry fitted statistics into inference",
                instructions: LearningExamples.frozenStandardizationReference.instructions, probes: [],
                allowedNodeTypes: ["input", "linear", "add", "concat", "output"])
        }
        if identity == .backwardChain {
            return AffineLessonTask(identity:identity,title:"Backpropagate through an affine chain",instructions:LearningExamples.affineBackwardReference.instructions,probes:[],allowedNodeTypes:["input","linear","add","concat","output"])
        }
        if let mode=identity.geluApproximateTanh {
            return ModelConstructionLessonTask(identity:identity,title:mode ? "Construct tanh-approximate GELU":"Construct exact GELU",instructions:LearningExamples.geluReference(approximateTanh:mode).instructions,probes:[],allowedNodeTypes:["input","linear","gelu","add","concat","output"])
        }
        let target = target(identity.taskSeed!)
        let probes = zip(["zero", "basis-0", "basis-1"], [[0.0,0.0],[1.0,0.0],[0.0,1.0]]).map {
            AffineProbe(id: $0.0, input: $0.1, output: apply($0.1, target: target))
        }
        return AffineLessonTask(identity: identity, title: "Recover an affine map from its behavior",
            instructions: "Construct an affine graph matching the observed outputs at zero and the two basis vectors. Predict how bias and matrix columns determine these observations. Equivalent affine constructions are accepted; nonlinear activations are outside this exercise.",
            probes: probes, allowedNodeTypes: ["input", "linear", "add", "concat", "output"])
    }
    public static func plan(identity: AffineLessonIdentity, runtimeVersion: String, fixtureSeed: UInt64,
                            assessmentVersion: ContractVersion? = nil) throws -> LearningAssessmentPlan {
        try identity.validate()
        if identity == .sigmoid {
            guard assessmentVersion == nil || assessmentVersion == ContractVersion() else {
                throw LearningContractError("Unsupported Sigmoid assessment version.")
            }
            let reference = LearningExamples.sigmoidReference
            var generator = ProbeGenerator(state: fixtureSeed)
            let fixtures = try (0..<4).map { index -> LearningFixture in
                let negative = -(1.25 + Double(generator.next()%176)/100)
                let near = Double(generator.next()%101)/1000
                let positive = 1.25 + Double(generator.next()%176)/100
                // Repeated anchors with different companions distinguish elementwise
                // behavior from normalization across either rows or columns.
                let values: [Double]
                switch index {
                case 0: values = [0, negative, near, positive]
                case 1: values = [0, positive, negative, -near]
                case 2: values = [-8, near, positive, 8]
                default: values = [negative, -1, 1, positive]
                }
                return LearningFixture(id: "sigmoid-\(fixtureSeed)-\(index)",
                    inputs: ["x": .init(shape: [2,2], values: values)],
                    expected: ["output": .init(shape: [2,2], values: try values.map(reference.apply))])
            }
            return LearningAssessmentPlan(id: identity.assessmentID, objectiveID: "activations.sigmoid",
                task: reference.instructions, runtime: .init(backend: "pytorch", version: runtimeVersion,
                    capabilities: ["linear": ContractVersion(1,1,0), "sigmoid": ContractVersion()]),
                fixtureSeed: fixtureSeed, fixtures: fixtures,
                criteria: [.init(id: "sigmoid-output", description: "Transform each value independently with Sigmoid, preserving rows and feature order.",
                    nextStep: "Review Linear weights and offsets in Parameters. Preserve each input before Sigmoid; unlike Softmax, Sigmoid does not mix values across a direction.",
                    rule: .tensor(key: "output", absoluteTolerance: 1e-6, relativeTolerance: 1e-6))])
        }
        if identity == .tanh {
            guard assessmentVersion == nil || assessmentVersion == ContractVersion() else {
                throw LearningContractError("Unsupported Tanh assessment version.")
            }
            let reference = LearningExamples.tanhReference
            var generator = ProbeGenerator(state: fixtureSeed)
            let fixtures = try (0..<4).map { index -> LearningFixture in
                let negative = -(1.25 + Double(generator.next()%176)/100)
                let near = Double(generator.next()%101)/1000
                let positive = 1.25 + Double(generator.next()%176)/100
                // Repeated anchors with different companions distinguish elementwise
                // behavior from normalization across either rows or columns.
                let values: [Double]
                switch index {
                case 0: values = [0, negative, near, positive]
                case 1: values = [0, positive, negative, -near]
                case 2: values = [-8, near, positive, 8]
                default: values = [negative, -1, 1, positive]
                }
                return LearningFixture(id: "tanh-\(fixtureSeed)-\(index)",
                    inputs: ["x": .init(shape: [2,2], values: values)],
                    expected: ["output": .init(shape: [2,2], values: try values.map(reference.apply))])
            }
            return LearningAssessmentPlan(id: identity.assessmentID, objectiveID: "activations.tanh",
                task: reference.instructions, runtime: .init(backend: "pytorch", version: runtimeVersion,
                    capabilities: ["linear": ContractVersion(1,1,0), "tanh": ContractVersion(1,1,0)]),
                fixtureSeed: fixtureSeed, fixtures: fixtures,
                criteria: [.init(id: "tanh-output", description: "Transform each value independently with Tanh, preserving rows and feature order.",
                    nextStep: "Review Linear weights and offsets in Parameters. Preserve each input before Tanh; unlike Softmax, Tanh does not mix values across a direction.",
                    rule: .tensor(key: "output", absoluteTolerance: 1e-6, relativeTolerance: 1e-6))])
        }
        if identity == .relu {
            guard assessmentVersion == nil || assessmentVersion == ContractVersion() else {
                throw LearningContractError("Unsupported ReLU assessment version.")
            }
            let reference = LearningExamples.reluReference
            var generator = ProbeGenerator(state: fixtureSeed)
            let fixtures = try (0..<4).map { index -> LearningFixture in
                let negative = -(1.25 + Double(generator.next()%176)/100)
                let near = Double(generator.next()%101)/1000
                let positive = 1.25 + Double(generator.next()%176)/100
                // Repeated anchors with different companions distinguish elementwise
                // behavior from normalization across either rows or columns.
                let values: [Double]
                switch index {
                case 0: values = [0, negative, near, positive]
                case 1: values = [0, positive, negative, -near]
                case 2: values = [-100, -near, positive, 100]
                default: values = [negative, -1, 1, positive]
                }
                return LearningFixture(id: "relu-\(fixtureSeed)-\(index)",
                    inputs: ["x": .init(shape: [2,2], values: values)],
                    expected: ["output": .init(shape: [2,2], values: try values.map(reference.apply))])
            }
            return LearningAssessmentPlan(id: identity.assessmentID, objectiveID: "activations.relu",
                task: reference.instructions, runtime: .init(backend: "pytorch", version: runtimeVersion,
                    capabilities: ["linear": ContractVersion(1,1,0), "relu": ContractVersion(1,1,0)]),
                fixtureSeed: fixtureSeed, fixtures: fixtures,
                criteria: [.init(id: "relu-output", description: "Transform each value independently with ReLU, preserving rows and feature order.",
                    nextStep: "Review Linear weights and offsets in Parameters. Preserve each input before ReLU; unlike Softmax, ReLU does not mix values across a direction.",
                    rule: .tensor(key: "output", absoluteTolerance: 1e-6, relativeTolerance: 1e-6))])
        }
        if identity == .composition {
            return try LearningExamples.affineCompositionForReplay(runtimeVersion: runtimeVersion, seed: fixtureSeed,
                assessmentVersion: assessmentVersion ?? ContractVersion(1,1,0), contentVersion: identity.contentVersion)
        }
        if let mode=identity.geluApproximateTanh {
            guard assessmentVersion == nil || assessmentVersion == ContractVersion() else {throw LearningContractError("Unsupported GELU assessment version.")}
            let reference=LearningExamples.geluReference(approximateTanh:mode)
            var generator=ProbeGenerator(state:fixtureSeed)
            let fixtures=try (0..<4).map { index -> LearningFixture in
                let negative = -(1.25 + Double(generator.next()%176)/100)
                let near = 0.001 + Double(generator.next()%100)/1000
                let positiveNear = 0.001 + Double(generator.next()%100)/1000
                let positive = 1.25 + Double(generator.next()%176)/100
                // Guarantee a mode-separating coordinate while retaining seed variation elsewhere.
                let values=[negative,index == 0 ? 0:-near,positiveNear,index == 0 ? 2.5:positive]
                return LearningFixture(id:"gelu-\(fixtureSeed)-\(index)",inputs:["x":LearningTensor(shape:[2,2],values:values)],expected:["output":LearningTensor(shape:[2,2],values:try values.map(reference.apply))])
            }
            return LearningAssessmentPlan(id:identity.assessmentID,objectiveID:mode ? "activations.gelu-tanh":"activations.gelu-exact",task:reference.taskText,
                runtime:LearningRuntime(backend:"pytorch",version:runtimeVersion,capabilities:["linear":ContractVersion(1,1,0),"gelu":ContractVersion()]),fixtureSeed:fixtureSeed,fixtures:fixtures,
                criteria:[LearningCriterion(id:"gelu-output",description:"Match elementwise GELU on balanced negative, zero, near-zero and positive fixtures.",nextStep:"Check the GELU approximation setting and your surrounding Linear parameters. Preserve negative GELU values; neither ReLU clipping nor Tanh alone gives this function.",rule:.tensor(key:"output",absoluteTolerance:1e-6,relativeTolerance:1e-6))])
        }
        if identity == .frozenStandardization {
            guard assessmentVersion == nil || assessmentVersion == ContractVersion() else {
                throw LearningContractError("Unsupported frozen-standardization assessment version.")
            }
            let reference = LearningExamples.frozenStandardizationReference
            var generator = ProbeGenerator(state: fixtureSeed)
            let fixtures = try (0..<4).map { index -> LearningFixture in
                var values = (0..<4).map { _ in generator.input() }
                // Repeat an anchor in distinct batches, including a distant companion.
                if index == 0 || index == 3 { values[0] = 2; values[1] = 12 }
                if index == 3 { values[2] += 200; values[3] -= 400 }
                let output = try reference.apply(Array(values[0..<2])) + reference.apply(Array(values[2..<4]))
                return LearningFixture(id: "frozen-standardization-\(fixtureSeed)-\(index)",
                    inputs: ["x": .init(shape: [2,2], values: values)], expected: ["output": .init(shape: [2,2], values: output)])
            }
            return LearningAssessmentPlan(id: identity.assessmentID, objectiveID: "data.frozen-standardization-as-affine",
                task: reference.instructions, runtime: .init(backend: "pytorch", version: runtimeVersion,
                    capabilities: ["linear": ContractVersion(1,1,0)]), fixtureSeed: fixtureSeed, fixtures: fixtures,
                criteria: [LearningCriterion(id: "frozen-transform", description: "Apply saved means and scales to new observations, preserving feature order and batch independence.",
                    nextStep: "Use diagonal weights 1/s and bias −μ/s in row-batch Linear convention. Do not compute statistics from these new inputs or omit the offset.",
                    rule: .tensor(key: "output", absoluteTolerance: 1e-5, relativeTolerance: 1e-5))])
        }
        if identity == .backwardChain {
            guard assessmentVersion == nil || assessmentVersion == ContractVersion() else {throw LearningContractError("Unsupported backward-chain assessment version.")}
            let reference=LearningExamples.affineBackwardReference
            var generator=ProbeGenerator(state:fixtureSeed)
            let fixtures=try (0..<4).map { index -> LearningFixture in
                let values=(0..<4).map{_ in generator.input()}
                let output=try reference.apply(Array(values[0..<2])) + reference.apply(Array(values[2..<4]))
                return LearningFixture(id:"backward-\(fixtureSeed)-\(index)",inputs:["x":LearningTensor(shape:[2,2],values:values)],expected:["output":LearningTensor(shape:[2,2],values:output)])
            }
            return LearningAssessmentPlan(id:identity.assessmentID,objectiveID:"matrices.affine-backward-chain",task:reference.taskText,
                runtime:LearningRuntime(backend:"pytorch",version:runtimeVersion,capabilities:["linear":ContractVersion(1,1,0)]),fixtureSeed:fixtureSeed,fixtures:fixtures,
                criteria:[LearningCriterion(id:"input-gradient",description:"Match input gradients for four independent upstream-gradient fixtures.",nextStep:"Reverse the forward traversal. In row convention multiply g by W2 then W1; Linear weights must be transposed. Bias contributes no input-gradient shift.",rule:.tensor(key:"output",absoluteTolerance:1e-5,relativeTolerance:1e-5))])
        }
        guard assessmentVersion == nil || assessmentVersion == ContractVersion() else {
            throw LearningContractError("Unsupported affine recovery assessment version.")
        }
        let task = try publicTask(identity: identity)
        let target = target(identity.taskSeed!)
        var generator = ProbeGenerator(state: fixtureSeed)
        let fixtures = (0..<4).map { index -> LearningFixture in
            let values = (0..<4).map { _ in generator.input() }
            let output = apply(Array(values[0..<2]), target: target) + apply(Array(values[2..<4]), target: target)
            return LearningFixture(id: "probe-\(fixtureSeed)-\(index)", inputs: ["x": LearningTensor(shape: [2,2], values: values)],
                expected: ["output": LearningTensor(shape: [2,2], values: output)])
        }
        let observations = task.probes.map { "\($0.id): f(\($0.input))=\($0.output)" }.joined(separator: "; ")
        return LearningAssessmentPlan(id: identity.assessmentID, objectiveID: "matrices.affine-from-probes",
            task: task.instructions + " " + observations,
            runtime: LearningRuntime(backend: "pytorch", version: runtimeVersion, capabilities: ["linear": ContractVersion(1,1,0)]),
            fixtureSeed: fixtureSeed, fixtures: fixtures,
            criteria: [LearningCriterion(id: "affine-output", description: "Match measured outputs on four independent two-row fixtures.",
                nextStep: "The output at zero gives the bias. Subtract it from each basis-vector output to recover the corresponding matrix column.",
                rule: .tensor(key: "output", absoluteTolerance: 1e-5, relativeTolerance: 1e-5))])
    }
}
private struct ProbeGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return state >> 32
    }
    mutating func coefficient() -> Double { Double(Int(next() % 13) - 6) / 2 }
    mutating func input() -> Double { Double(Int(next() % 2001) - 1000) / 100 }
}
