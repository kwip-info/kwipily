import Foundation

/// Scalar comparisons, residuals, normalization and learned tables; no native attention gold.
public struct AttentionLessonReference {
    public let expected: [String: LearningTensor]
    public init(inputs: [String: LearningTensor], task: AttentionLearningTask) throws {
        let keys = Set(task.sourceRoles + task.parameterShapes.keys.map { "parameter."+$0 } + task.outputRoles.map { "upstream."+$0 })
        guard Set(inputs.keys) == keys, let n = inputs[task.sourceRoles[0]]?.shape.first, (1...3).contains(n), inputs.values.allSatisfy(\.wellFormed) else { throw LearningContractError("Use the supplied attention arrays, learned tables and result influence.") }
        for role in task.sourceRoles { guard inputs[role]!.shape == [n]+task.sampleShapes[role]! else { throw LearningContractError("A supplied sequence has the wrong observation or position shape.") } }
        for (name, shape) in task.parameterShapes { guard inputs["parameter."+name]!.shape == shape else { throw LearningContractError("A checker-owned learned table has the wrong shape.") } }
        for role in task.outputRoles { guard inputs["upstream."+role]!.shape == [n]+task.outputSampleShapes[role]! else { throw LearningContractError("Result influence must match its named output.") } }
        var answer = Self.forward(inputs, task: task), scratch = inputs
        func weighted() -> Double { let output = Self.forward(scratch, task: task); return task.outputRoles.reduce(0) { total, role in total + zip(output[role]!.values, inputs["upstream."+role]!.values).reduce(0) { $0+$1.0*$1.1 } } }
        for key in task.sourceRoles + task.parameterShapes.keys.sorted().map({ "parameter."+$0 }) {
            var gradients: [Double] = []
            for i in inputs[key]!.values.indices {
                let value=inputs[key]!.values[i]
                func central(_ step:Double)->Double {
                    scratch[key]!.values[i]=value+step;let plus=weighted()
                    scratch[key]!.values[i]=value-step;let minus=weighted()
                    scratch[key]!.values[i]=value
                    return (plus-minus)/(2*step)
                }
                // Two independent scalar central differences cancel their leading
                // quadratic truncation error. Width-two normalization can amplify
                // that error; the native worker and grading tolerances stay fixed.
                func refined(_ coarse:Double,_ fine:Double)->Double {fine+(fine-coarse)/3}
                if task == .scaled {
                    // Direct scaled attention has no normalization stack. Prefer
                    // larger perturbations only when two extrapolations agree;
                    // this avoids subtractive roundoff for a linear value path.
                    let coarse=central(1e-4),fine=central(5e-5),finer=central(2.5e-5)
                    let estimate=refined(coarse,fine),check=refined(fine,finer)
                    if abs(estimate-check)<=1e-9+1e-9*abs(check) {
                        gradients.append(estimate);continue
                    }
                }
                let coarse=central(1e-6),fine=central(5e-7)
                gradients.append(refined(coarse,fine))
            }
            answer[key+"Gradient"] = .init(shape: inputs[key]!.shape, values: gradients)
        }
        answer.merge(task.expectedCheckpoints(observations:n)) {_,new in new}
        expected = answer
    }
    private static func forward(_ f: [String: LearningTensor], task: AttentionLearningTask) -> [String: LearningTensor] {
        typealias Rows = [[Double]]
        let n = f[task.sourceRoles[0]]!.shape[0]
        func p(_ key: String) -> [Double] { f["parameter."+key]!.values }
        func linear(_ x: Rows, _ w: [Double], _ b: [Double]) -> Rows { x.map { row in b.indices.map { j in row.indices.reduce(b[j]) { $0 + w[j*row.count+$1]*row[$1] } } } }
        func normalize(_ x: Rows, _ prefix: String) -> Rows {
            let weight = p(prefix+"weight"), bias = p(prefix+"bias")
            return x.map { row in let mean = row.reduce(0,+)/Double(row.count); let variance = row.reduce(0) { $0+($1-mean)*($1-mean) }/Double(row.count); return row.indices.map { (row[$0]-mean)/sqrt(variance+1e-5)*weight[$0]+bias[$0] } }
        }
        func add(_ a: Rows, _ b: Rows) -> Rows { zip(a,b).map { ar, br in zip(ar,br).map { $0.0+$0.1 } } }
        func attention(_ query: Rows, _ key: Rows, _ value: Rows, _ prefix: String = "", blocked: Bool = false, causal: Bool = false, learned: Bool = true) -> (Rows, [Double]) {
            var q = query, k = key, v = value
            if learned { let w = p(prefix+"in_proj_weight"), b = p(prefix+"in_proj_bias"); q = linear(q, Array(w[0..<4]), Array(b[0..<2])); k = linear(k, Array(w[4..<8]), Array(b[2..<4])); v = linear(v, Array(w[8..<12]), Array(b[4..<6])) }
            var output: Rows = [], weights: [Double] = []
            for (i, row) in q.enumerated() {
                let scores = k.map { kr in zip(row,kr).reduce(0) { $0+$1.0*$1.1 }/sqrt(2) }
                let allowed = k.indices.map { !((blocked && i == 0 && $0 == 1) || (causal && $0 > i)) }
                let high = zip(scores,allowed).filter { $0.1 }.map { $0.0 }.max()!
                let expScores = zip(scores,allowed).map { $0.1 ? exp($0.0-high) : 0 }, total = expScores.reduce(0,+)
                let shares = expScores.map { $0/total }; weights += shares
                output.append((0..<2).map { d in v.indices.reduce(0) { $0+shares[$1]*v[$1][d] } })
            }
            if learned { output = linear(output, p(prefix+"out_proj.weight"), p(prefix+"out_proj.bias")) }
            return (output,weights)
        }
        func feedforward(_ x: Rows, _ prefix: String) -> Rows { let hidden = linear(x,p(prefix+"linear1.weight"),p(prefix+"linear1.bias")).map { $0.map { 0.5*$0*(1+erf($0/sqrt(2))) } }; return linear(hidden,p(prefix+"linear2.weight"),p(prefix+"linear2.bias")) }
        func encode(_ x: Rows, _ prefix: String, blocked: Bool = false) -> Rows { let a = normalize(x,prefix+"norm1."); let y = add(x,attention(a,a,a,prefix+"self_attn.",blocked:blocked).0); return add(y,feedforward(normalize(y,prefix+"norm2."),prefix)) }
        func decode(_ x: Rows, _ memory: Rows, _ prefix: String) -> Rows { let a = normalize(x,prefix+"norm1."); let y = add(x,attention(a,a,a,prefix+"self_attn.",causal:true).0); let z = add(y,attention(normalize(y,prefix+"norm2."),memory,memory,prefix+"multihead_attn.",blocked:true).0); return add(z,feedforward(normalize(z,prefix+"norm3."),prefix)) }
        var result: [Double] = [], weights: [Double] = []
        for observation in 0..<n {
            func source(_ role: String) -> Rows { let length = task.sampleShapes[role]!.reduce(1,*)/2, values = f[role]!.values; return (0..<length).map { t in let offset = (observation*length+t)*2; return Array(values[offset..<offset+2]) } }
            let output: Rows
            switch task {
            case .multihead, .scaled: let pair = attention(source("query"),source("key"),source("value"),blocked:true,learned:task == .multihead); output = pair.0; weights += pair.1
            case .encoder: output = normalize(encode(source("in"),"layers.0.",blocked:true),"norm.")
            case .decoder: output = normalize(decode(source("target"),source("memory"),"layers.0."),"norm.")
            case .transformer: let memory = normalize(encode(source("source"),"encoder.layers.0."),"encoder.norm."); output = normalize(decode(source("target"),memory,"decoder.layers.0."),"decoder.norm.")
            }
            result += output.flatMap { $0 }
        }
        var answer = ["result": LearningTensor(shape: [n]+task.outputSampleShapes["result"]!, values: result)]
        if task == .multihead { answer["weights"] = .init(shape: [n]+task.outputSampleShapes["weights"]!, values: weights) }
        return answer
    }
}
public enum AttentionAssessment {
    public static func plan(task: AttentionLearningTask, runtimeVersion: String, seed: UInt64) throws -> LearningAssessmentPlan {
        var state = seed
        func tensor(_ shape: [Int], normalization: Bool = false) -> LearningTensor { .init(shape: shape, values: (0..<shape.reduce(1,*)).map { _ in state = state &* 6364136223846793005 &+ 1442695040888963407; let value = Double(Int((state>>32)%19)-9)/20; return normalization ? 1+value/5 : value }) }
        let fixtures = try (0..<4).map { index -> LearningFixture in
            let n = index%3+1; var inputs: [String: LearningTensor] = [:]
            for role in task.sourceRoles { inputs[role] = tensor([n]+task.sampleShapes[role]!) }
            for name in task.parameterShapes.keys.sorted() { inputs["parameter."+name] = tensor(task.parameterShapes[name]!, normalization: name.contains("norm") && name.hasSuffix("weight")) }
            for role in task.outputRoles { inputs["upstream."+role] = tensor([n]+task.outputSampleShapes[role]!) }
            return .init(id: "\(task.rawValue)-\(seed)-\(index)", inputs: inputs, expected: try AttentionLessonReference(inputs: inputs, task: task).expected)
        }
        let criteria = fixtures[0].expected.keys.sorted().map { key in LearningCriterion(id: key, description: key == "blockedConnections" ? "Block query position 0 from key position 1; keep every other connection allowed." : key == "blockedSourcePositions" ? "Keep every supplied source position available." : key.hasSuffix("Gradient") ? "Measure how the supplied sequences and learned tables affect the result." : key == "weights" ? "Show the share assigned to each allowed source position." : "Mix the allowed values for each observation and requested position.", nextStep: task.controlInstructions, rule: .tensor(key: key, absoluteTolerance: task.checkpointSampleShapes[key] != nil ? 0:3e-6, relativeTolerance: task.checkpointSampleShapes[key] != nil ? 0:3e-5)) }
        let plan = LearningAssessmentPlan(id: "assessment."+task.rawValue, objectiveID: "attention."+task.rawValue, task: task.instructions, runtime: .init(backend: "pytorch", version: runtimeVersion, capabilities: [task.componentID: ContractVersion(), "attention_mask": ContractVersion()]), fixtureSeed: seed, fixtures: fixtures, criteria: criteria)
        try plan.validate(); return plan
    }
}
