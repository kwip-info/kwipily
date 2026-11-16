import Foundation

/// Independent source selection and grouping arithmetic. A padded cell is display
/// metadata only; batchLengths identifies the actual observations in each group.
public struct DataLoadingLessonReference {
    public let expected:[String:LearningTensor]
    public init(task:DataLoadingLearningTask,inputs:[String:LearningTensor])throws {
        guard let rows=inputs["sourceRows"],let targets=inputs["desiredRows"],let ids=inputs["rowIDs"],rows.wellFormed,targets.wellFormed,ids.wellFormed,
              let count=rows.shape.first,(5...32).contains(count),targets.shape==[count,1],ids.shape==[count],Set(ids.values).count==count,
              ids.values.allSatisfy({$0>=0 && $0<=1_000_000 && $0.rounded(.towardZero)==$0}),rows.values.allSatisfy({abs($0)<=1_000_000}),targets.values.allSatisfy({abs($0)<=1_000_000}) else {throw LearningContractError("Supply aligned finite source rows, desired outcomes and distinct whole-number observation IDs.")}
        let shape=task == .collation ? [count,3,1]:task == .continuation ? [count,3]:[count,2]
        guard rows.shape==shape else {throw LearningContractError("Keep the supplied row and sequence directions for this lesson.")}
        if task == .continuation {
            guard let lengths=inputs["sourceLengths"],let weights=inputs["sourceWeights"],lengths.wellFormed,weights.wellFormed,lengths.shape==[count],weights.shape==[count,3],lengths.values.allSatisfy({(0...3).contains($0) && $0.rounded(.towardZero)==$0}),rows.values.allSatisfy({(0...15).contains($0) && $0.rounded(.towardZero)==$0}) else {throw LearningContractError("Whole-group continuation needs real category rows, used-item counts and aligned item weights.")}
        }
        if task == .collation {
            guard let lengths=inputs["sourceLengths"],lengths.wellFormed,lengths.shape==[count],lengths.values.allSatisfy({(0...3).contains($0) && $0.rounded(.towardZero)==$0}) else {throw LearningContractError("Keep each padded sequence’s true whole-number length with its observation.")}
        }
        let sourceTasks:[DataLoadingLearningTask]=[.indexed,.tensor,.subset,.iterable]
        if let sourceKind=sourceTasks.firstIndex(of:task) {
            let selected=task == .subset ? [4,1,0]:Array(0..<(task == .iterable ? 5:count))
            var selectedRows:[Double]=[]
            for row in selected {selectedRows += Array(rows.values[(row*2)..<(row*2+2)])}
            expected=["selectedRowIDs":.init(shape:[selected.count],values:selected.map {ids.values[$0]}),"selectedInputs":.init(shape:[selected.count,2],values:selectedRows),"selectedTargets":.init(shape:[selected.count,1],values:selected.map {targets.values[$0]}),"sourceKind":.init(shape:[1],values:[Double(sourceKind)])]
        } else if task == .shuffle {
            expected=["shuffleChecks":.init(shape:[4],values:[1,1,1,1]),"selectionDifference":.init(shape:[count],values:Array(repeating:0,count:count)),"alignmentDifference":.init(shape:[count,3],values:Array(repeating:0,count:count*3))]
        } else if task == .continuation {
            expected=["continuationChecks":.init(shape:[4],values:[1,1,1,1]),"readerPolicy":.init(shape:[2],values:[2,1]),"parentByteDifference":.init(shape:[1],values:[0]),"epochOrderDifference":.init(shape:[count],values:Array(repeating:0,count:count))]
        } else {
            let batch=4,groups=(count+batch-1)/batch,width=task == .collation ? 3:2
            var batchedIDs=[Double](repeating:-1,count:groups*batch),batchedTargets=[Double](repeating:0,count:groups*batch),batchedRows=[Double](repeating:0,count:groups*batch*width),lengths:[Double]=[]
            var sequenceLengths=[Double](repeating:0,count:groups*batch)
            for group in 0..<groups {lengths.append(Double(min(batch,count-group*batch)))}
            for row in 0..<count {
                batchedIDs[row]=ids.values[row];batchedTargets[row]=targets.values[row]
                for direction in 0..<width {batchedRows[row*width+direction]=rows.values[row*width+direction]}
                if task == .collation {sequenceLengths[row]=inputs["sourceLengths"]!.values[row]}
            }
            var result:[String:LearningTensor]=["batchRowIDs":.init(shape:[groups,batch],values:batchedIDs),"batchTargets":.init(shape:[groups,batch],values:batchedTargets)]
            if task == .batches {
                result["batchLengths"] = .init(shape:[groups],values:lengths)
                let complete=count/batch
                result["dropLastRowIDs"] = .init(shape:[complete,batch],values:Array(ids.values.prefix(complete*batch)))
            } else {
                result["batchedInputs"] = .init(shape:[groups,batch,3,1],values:batchedRows)
                result["sequenceLengths"] = .init(shape:[groups,batch],values:sequenceLengths)
            }
            expected=result
        }
    }
    public static func label(_ key:String)->String {
        switch key {case "sourceKind":return "Observed native source kind";case "selectedRowIDs":return "Selected observation IDs";case "selectedInputs":return "Selected input values";case "selectedTargets":return "Aligned desired outcomes";case "batchRowIDs":return "Observation IDs in each group";case "batchLengths":return "Actual group sizes";case "batchTargets":return "Desired outcomes in each group";case "dropLastRowIDs":return "Groups when the short ending is dropped";case "shuffleChecks":return "Mixing repeat, changed seed, membership and alignment";case "selectionDifference":return "Observation membership difference";case "alignmentDifference":return "Input and desired-outcome alignment difference";case "batchedInputs":return "Padded sequence groups";case "sequenceLengths":return "True lengths in each group";case "continuationChecks":return "Actual resumed order, settings, memory and worker cleanup";case "readerPolicy":return "Observed reader processes and prefetch";case "parentByteDifference":return "Saved parent byte difference";case "epochOrderDifference":return "Resumed observation-order difference";default:return key}
    }
}
public enum DataLoadingAssessment {
    public static func plan(task:DataLoadingLearningTask,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        var state=seed
        func draw()->Double {state=state &* 6364136223846793005 &+ 1442695040888963407;return Double(Int((state>>32)%33)-16)/16}
        var fixtures:[LearningFixture]=[]
        for (variant,count) in [9,10,11,13].enumerated() {
            var rowValues:[Double]=[],targetValues:[Double]=[],ids:[Double]=[],lengths:[Double]=[],weights:[Double]=[]
            for row in 0..<count {
                ids.append(Double(variant*100+row));targetValues.append(draw())
                let length=(row+variant)%4;lengths.append(Double(length))
                if task == .collation {for position in 0..<3 {rowValues.append(position<length ? draw():0)}}
                else if task == .continuation {for position in 0..<3 {rowValues.append(Double((row+position+variant)%11));weights.append(draw())}}
                else {rowValues.append(draw());rowValues.append(draw())}
            }
            let shape=task == .collation ? [count,3,1]:task == .continuation ? [count,3]:[count,2]
            var inputs:[String:LearningTensor]=["sourceRows":.init(shape:shape,values:rowValues),"desiredRows":.init(shape:[count,1],values:targetValues),"rowIDs":.init(shape:[count],values:ids),"seed":.init(shape:[1],values:[Double((state>>32)%4_294_967_296)])]
            if task == .collation || task == .continuation {inputs["sourceLengths"] = .init(shape:[count],values:lengths)}
            if task == .continuation {inputs["sourceWeights"] = .init(shape:[count,3],values:weights)}
            fixtures.append(.init(id:task.rawValue+"-\(seed)-\(variant)",inputs:inputs,expected:try DataLoadingLessonReference(task:task,inputs:inputs).expected))
        }
        let criteria=fixtures[0].expected.keys.sorted().map {key in LearningCriterion(id:key,description:DataLoadingLessonReference.label(key),nextStep:task.controlInstructions,rule:.tensor(key:key,absoluteTolerance:0,relativeTolerance:0))}
        let capabilities=Dictionary(uniqueKeysWithValues:task.componentIDs.map {($0,ContractVersion())})
        let plan=LearningAssessmentPlan(id:"assessment."+task.rawValue,objectiveID:"data-loading."+task.rawValue,task:task.instructions,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:capabilities),budget:.init(milliseconds:60_000,maximumTensorElements:100_000),fixtureSeed:seed,fixtures:fixtures,criteria:criteria)
        try validate(plan);return plan
    }
    public static func validate(_ plan:LearningAssessmentPlan)throws {
        guard let task=DataLoadingLearningTask.allCases.first(where:{plan.id=="assessment."+$0.rawValue}),plan.objectiveID=="data-loading."+task.rawValue,plan.contentVersion==ContractVersion(),plan.assessmentVersion==ContractVersion(),plan.runtime.backend=="pytorch",plan.runtime.capabilities==Dictionary(uniqueKeysWithValues:task.componentIDs.map {($0,ContractVersion())}),plan.fixtures.count==4 else {throw LearningContractError("Use the current four-fixture data loading assessment and its independently required capabilities.")}
        try plan.validate()
        for fixture in plan.fixtures {
            let expected=try DataLoadingLessonReference(task:task,inputs:fixture.inputs).expected
            guard fixture.expected==expected,Set(plan.criteria.map(\.id))==Set(expected.keys) else {throw LearningContractError("Keep independent source, alignment and grouping expectations for every private fixture.")}
            for criterion in plan.criteria {guard case .tensor(let key,let absolute,let relative)=criterion.rule,key==criterion.id,absolute==0,relative==0 else {throw LearningContractError("Data grouping checks require exact observed IDs, alignment and policy outcomes.")}}
        }
    }
}
