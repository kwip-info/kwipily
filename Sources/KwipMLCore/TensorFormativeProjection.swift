import Foundation

/// View-only values. Assessment-origin receipts never expose their hidden fixtures.
public struct TensorFormativeFixture:Equatable,Identifiable {
    public let id:String
    public let input:LearningTensor
    public let upstream:LearningTensor
    public let measured:[String:LearningTensor]
    public let expected:[String:LearningTensor]
    public var inputLabel:String = "Input x"
    public var sourceLayout:String? = nil
    public var sourceLengths:LearningTensor? = nil
    public var sourceWeights:LearningTensor? = nil
    public var sourceTable:LearningTensor? = nil
    public var secondaryInput:LearningTensor? = nil
    public var sharedParameters:[String:LearningTensor] = [:]
    public var secondaryInputLabel:String = "Right operand B"
}
public struct TensorFormativeProjection:Equatable {
    public let submissionDigest:String
    public let fixtures:[TensorFormativeFixture]
}
public extension TensorLearningReceipt {
    func formativeProjection(expectedProjectID:String,currentSubmission:Data,currentMode:LearningMode)->TensorFormativeProjection? {
        guard currentMode != .assessment,attempt.mode != .assessment,currentSubmission == submissionData,
              let plan=try? verifiedReplay(expectedProjectID:expectedProjectID) else {return nil}
        let bagSubmission=try? EmbeddingBagLearningSubmission.decode(submissionData)
        let bagSource=bagSubmission?.graph.nodes.first {$0.type == "bag_input"}
        let bagConfiguration=bagSource.flatMap {try? BagSourceConfiguration(values:$0.config)}
        var fixtures:[TensorFormativeFixture]=[]
        for fixture in plan.fixtures {
            guard let measured=evidence.fixtures.first(where:{$0.fixtureID == fixture.id}),
                  measured.tensors.values.allSatisfy(\.wellFormed),
                  let input=fixture.inputs["sourceRows"] ?? fixture.inputs["x"] ?? fixture.inputs["indices"] ?? fixture.inputs["left"],let upstream=fixture.inputs["upstream"] else {return nil}
            let label=bagConfiguration != nil ? "Category IDs by observation group" : fixture.inputs["left"] != nil ? "Left operand A" : fixture.inputs["indices"] == nil ? "Input x" : "Integer selectors"
            fixtures.append(.init(id:fixture.id,input:input,upstream:upstream,measured:measured.tensors,expected:fixture.expected,inputLabel:label,sourceLayout:bagConfiguration?.layout,sourceLengths:bagConfiguration != nil ? fixture.inputs["lengths"]:nil,sourceWeights:bagConfiguration != nil ? fixture.inputs["sourceWeights"]:nil,sourceTable:fixture.inputs["table"],secondaryInput:fixture.inputs["right"],sharedParameters:fixture.inputs.filter { ["weight","bias"].contains($0.key) }))
        }
        return .init(submissionDigest:submissionSHA256,fixtures:fixtures)
    }
}
public extension LearningTensor {
    /// Preserve the remaining coordinates when selecting one observation from axis zero.
    func batchObservation(at index:Int)->LearningTensor? {
        guard wellFormed,shape.count>=2,index>=0,index<shape[0] else {return nil}
        let count=values.count/shape[0],start=index*count
        return .init(shape:Array(shape.dropFirst()),values:Array(values[start..<(start+count)]))
    }
}

/// Recipe and artifact exercises have real source fields, without a synthetic upstream array.
public struct RecipeFormativeFixture:Identifiable,Equatable {
    public let id:String
    public let inputs:[String:LearningTensor]
    public let measured:[String:LearningTensor]
    public let expected:[String:LearningTensor]
}
public struct RecipeFormativeProjection:Equatable {
    public let operation:String
    public let submissionDigest:String
    public let fixtures:[RecipeFormativeFixture]
}
public extension TensorLearningReceipt {
    func recipeFormativeProjection(expectedProjectID:String,currentSubmission:Data,currentMode:LearningMode)->RecipeFormativeProjection? {
        guard DataLoadingLearningTask(rawValue:operation) != nil || ProgramExportLearningTask(rawValue:operation) != nil,
              currentMode != .assessment,attempt.mode != .assessment,currentSubmission==submissionData,
              let plan=try? verifiedReplay(expectedProjectID:expectedProjectID) else {return nil}
        var fixtures:[RecipeFormativeFixture]=[]
        for fixture in plan.fixtures {
            guard let measured=evidence.fixtures.first(where:{$0.fixtureID==fixture.id}),
                  Set(measured.tensors.keys)==Set(fixture.expected.keys),
                  fixture.inputs.values.allSatisfy(\.wellFormed),fixture.expected.values.allSatisfy(\.wellFormed),
                  measured.tensors.allSatisfy({$0.value.wellFormed && $0.value.shape==fixture.expected[$0.key]?.shape}) else {return nil}
            fixtures.append(.init(id:fixture.id,inputs:fixture.inputs,measured:measured.tensors,expected:fixture.expected))
        }
        return .init(operation:operation,submissionDigest:submissionSHA256,fixtures:fixtures)
    }
}
