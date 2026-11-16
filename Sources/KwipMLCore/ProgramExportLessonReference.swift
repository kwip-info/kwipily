import Foundation

/// Plain affine arithmetic and desired boundary outcomes, independent of export.
/// The trusted worker must measure isolation/tamper/continuation checks by doing
/// the real operations; these expected check values never fabricate evidence.
public struct ProgramExportLessonReference {
    public let expected:[String:LearningTensor]
    public init(task:ProgramExportLearningTask,inputs:[String:LearningTensor])throws {
        guard let rows=inputs["sourceRows"],let weight=inputs["parameter.weight"],let bias=inputs["parameter.bias"],rows.wellFormed,weight.wellFormed,bias.wellFormed,
              rows.shape.count==2,(1...4).contains(rows.shape[0]),rows.shape[1]==2,weight.shape==[2,2],bias.shape==[2],
              rows.values.allSatisfy({abs($0)<=16}),weight.values.allSatisfy({abs($0)<=16}),bias.values.allSatisfy({abs($0)<=16}) else {throw LearningContractError("Supply finite two-input, two-result rows and the actual small learned weight table and offsets.")}
        if task == .computation {guard rows.shape[0]==2 else {throw LearningContractError("The static example contains exactly two observations.")}}
        var output:[Double]=[]
        for row in 0..<rows.shape[0] {
            for result in 0..<2 {
                var value=bias.values[result]
                for input in 0..<2 {value += rows.values[row*2+input]*weight.values[result*2+input]}
                output.append(value)
            }
        }
        expected=["reloadedOutput":.init(shape:[rows.shape[0],2],values:output),"constraintChecks":.init(shape:[4],values:[1,1,1,1]),"artifactChecks":.init(shape:[3],values:[1,1,1]),"parentByteDifference":.init(shape:[1],values:[0])]
    }
    public static func label(_ key:String)->String {switch key {
    case "reloadedOutput":return "Actual reloaded program results"
    case "constraintChecks":return "Requested capture policy and accepted/rejected sizes"
    case "artifactChecks":return "Isolated reload, program pin and signature checks"
    case "parentByteDifference":return "Saved parent byte difference"
    default:return key
    }}
}
public enum ProgramExportAssessment {
    public static func plan(task:ProgramExportLearningTask,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        var state=seed
        func draw()->Double {state=state &* 6364136223846793005 &+ 1442695040888963407;return Double(Int((state>>32)%33)-16)/8}
        var fixtures:[LearningFixture]=[]
        for variant in 0..<4 {
            let count=task == .computation ? 2:[2,3,4,2][variant]
            var rows:[Double]=[],weights:[Double]=[],bias:[Double]=[]
            for _ in 0..<(count*2) {rows.append(draw())}
            for _ in 0..<4 {weights.append(draw())}
            for _ in 0..<2 {bias.append(draw())}
            let inputs:[String:LearningTensor]=["sourceRows":.init(shape:[count,2],values:rows),"parameter.weight":.init(shape:[2,2],values:weights),"parameter.bias":.init(shape:[2],values:bias)]
            fixtures.append(.init(id:task.rawValue+"-\(seed)-\(variant)",inputs:inputs,expected:try ProgramExportLessonReference(task:task,inputs:inputs).expected))
        }
        let criteria=fixtures[0].expected.keys.sorted().map {key in LearningCriterion(id:key,description:ProgramExportLessonReference.label(key),nextStep:task.controlInstructions,rule:.tensor(key:key,absoluteTolerance:key == "reloadedOutput" ? 2e-6:0,relativeTolerance:0))}
        let plan=LearningAssessmentPlan(id:"assessment."+task.rawValue,objectiveID:"program-export."+task.rawValue,task:task.instructions,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:[task.componentID:ContractVersion()]),budget:.init(milliseconds:60_000,maximumTensorElements:100_000),fixtureSeed:seed,fixtures:fixtures,criteria:criteria)
        try validate(plan);return plan
    }
    public static func validate(_ plan:LearningAssessmentPlan)throws {
        guard let task=ProgramExportLearningTask.allCases.first(where:{plan.id=="assessment."+$0.rawValue}),plan.objectiveID=="program-export."+task.rawValue,plan.contentVersion==ContractVersion(),plan.assessmentVersion==ContractVersion(),plan.runtime.backend=="pytorch",plan.runtime.capabilities==[task.componentID:ContractVersion()],plan.fixtures.count==4 else {throw LearningContractError("Use the current executable-program assessment and its exact independently required runtime capability.")}
        try plan.validate()
        for fixture in plan.fixtures {
            let expected=try ProgramExportLessonReference(task:task,inputs:fixture.inputs).expected
            guard fixture.expected==expected,Set(plan.criteria.map(\.id))==Set(expected.keys) else {throw LearningContractError("Preserve independent affine results and the complete actual export/reload checks.")}
            for criterion in plan.criteria {guard case .tensor(let key,let absolute,let relative)=criterion.rule,key==criterion.id,absolute==(key == "reloadedOutput" ? 2e-6:0),relative==0 else {throw LearningContractError("Use the reviewed numerical tolerance and exact artifact/boundary checks.")}}
        }
    }
}
