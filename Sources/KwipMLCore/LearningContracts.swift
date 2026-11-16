import Foundation
import CryptoKit

public enum LearningMode:String,Codable {case guided,practice,assessment}
public struct LearningObjective:Codable,Equatable,Identifiable {
    public var id:String;public var title:String;public var explanation:String;public var prerequisiteIDs:[String];public var version:ContractVersion
    public init(id:String,title:String,explanation:String,prerequisiteIDs:[String]=[],version:ContractVersion=ContractVersion()){self.id=id;self.title=title;self.explanation=explanation;self.prerequisiteIDs=prerequisiteIDs;self.version=version}
}
public struct LearningRuntime:Codable,Equatable {
    public var backend:String;public var version:String;public var capabilities:[String:ContractVersion]
    public init(backend:String,version:String,capabilities:[String:ContractVersion]){self.backend=backend;self.version=version;self.capabilities=capabilities}
}
public struct LearningBudget:Codable,Equatable {
    public var milliseconds:Double;public var maximumTensorElements:Int
    public init(milliseconds:Double=5000,maximumTensorElements:Int=4096){self.milliseconds=milliseconds;self.maximumTensorElements=maximumTensorElements}
}
public struct LearningTensor:Codable,Equatable {
    public var shape:[Int];public var values:[Double]
    public init(shape:[Int],values:[Double]){self.shape=shape;self.values=values}
    public var wellFormed:Bool {guard !shape.isEmpty,shape.count<=8,values.count<=4096,shape.allSatisfy({(1...4096).contains($0)}) else{return false};var count=1;for d in shape {guard count<=4096/d else{return false};count*=d};return count==values.count && values.allSatisfy(\.isFinite)}
}
/// Private grading material; the UI should expose task/criteria, never serialize this as a public lesson handout.
public struct LearningFixture:Codable,Equatable,Identifiable {
    public var id:String;public var inputs:[String:LearningTensor];public var expected:[String:LearningTensor]
    public init(id:String,inputs:[String:LearningTensor],expected:[String:LearningTensor]){self.id=id;self.inputs=inputs;self.expected=expected}
}
public enum LearningCriterionRule:Codable,Equatable {
    case tensor(key:String,absoluteTolerance:Double,relativeTolerance:Double)
    case trainingOnly(training:String,heldOut:String,fit:String)
}
public struct LearningCriterion:Codable,Equatable,Identifiable {
    public var id:String;public var description:String;public var nextStep:String;public var rule:LearningCriterionRule
    public init(id:String,description:String,nextStep:String,rule:LearningCriterionRule){self.id=id;self.description=description;self.nextStep=nextStep;self.rule=rule}
}
public struct LearningAssessmentPlan:Codable,Equatable,Identifiable {
    public var id:String;public var objectiveID:String;public var contentVersion:ContractVersion;public var assessmentVersion:ContractVersion
    public var task:String;public var runtime:LearningRuntime;public var budget:LearningBudget;public var fixtureSeed:UInt64;public var fixtures:[LearningFixture];public var criteria:[LearningCriterion]
    public init(id:String,objectiveID:String,contentVersion:ContractVersion=ContractVersion(),assessmentVersion:ContractVersion=ContractVersion(),task:String,runtime:LearningRuntime,budget:LearningBudget=LearningBudget(),fixtureSeed:UInt64,fixtures:[LearningFixture],criteria:[LearningCriterion]){self.id=id;self.objectiveID=objectiveID;self.contentVersion=contentVersion;self.assessmentVersion=assessmentVersion;self.task=task;self.runtime=runtime;self.budget=budget;self.fixtureSeed=fixtureSeed;self.fixtures=fixtures;self.criteria=criteria}
    public var fixtureDigest:String {let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys];guard let data=try? encoder.encode(fixtures) else{return "invalid"};return learningHash(data)}
    public func validate()throws {
        if let loss=LossLearningTask.allCases.first(where:{id == "assessment."+$0.rawValue}) {
            guard objectiveID == "loss."+loss.rawValue,contentVersion == ContractVersion(),assessmentVersion == ContractVersion(),runtime.backend == "pytorch",runtime.capabilities == [loss.componentID:ContractVersion()] else {throw LearningContractError("Loss fixture identity and capability must match the reviewed objective lesson.")}
            let keys=["trainingObjective","evaluationObjective","targetContract","reduction","inputInfluence"]
            let rules=Dictionary(uniqueKeysWithValues:zip(LossLearningAssessment.modes,keys))
            guard criteria.count == 5,Set(criteria.map(\.id)) == Set(rules.keys) else {throw LearningContractError("Loss lessons require all five independent objective criteria.")}
            for criterion in criteria {
                guard case .tensor(let key,let absolute,let relative)=criterion.rule,key == rules[criterion.id],absolute.isFinite,relative.isFinite,(0...1e-6).contains(absolute),(0...1e-5).contains(relative),key != "targetContract" || (absolute == 0 && relative == 0) else {throw LearningContractError("Loss criteria must use their own measured output and bounded numerical tolerance.")}
            }
            let templates=loss.fixtures(seed:0)[0].inputs
            for f in fixtures {
                guard Set(f.inputs.keys) == Set(loss.sourceRoles),Set(f.expected.keys) == Set(keys),let prediction=f.inputs["prediction"],let n=prediction.shape.first,(1...3).contains(n),prediction.shape == [n]+loss.predictionSampleShape else {throw LearningContractError("Loss fixtures require exact supplied roles and bounded observation-first prediction geometry.")}
                for role in loss.sourceRoles {
                    guard let tensor=f.inputs[role],tensor.shape == [n]+Array(templates[role]!.shape.dropFirst()),tensor.wellFormed else {throw LearningContractError("Loss target roles must preserve their reviewed sample geometry and observations.")}
                }
                let rawShape:[Int]=[LossLearningTask.crossEntropy,.nll,.ctc,.triplet].contains(loss) ? [n]:prediction.shape
                guard f.expected["trainingObjective"]?.shape == [1],f.expected["evaluationObjective"]?.shape == [1],f.expected["reduction"]?.shape == rawShape,f.expected["inputInfluence"]?.shape == prediction.shape else {throw LearningContractError("Loss combined scores, individual scores and raw-input influence need their distinct measured geometry.")}
                var format=prediction.shape.map(Double.init)
                if loss == .triplet {format += [0,1,2]}
                for role in loss.sourceRoles where role != "prediction" {
                    let tensor=f.inputs[role]!,integerRole=(loss == .crossEntropy || loss == .nll || loss == .ctc)
                    format += [integerRole ? 1:0,Double(tensor.shape.count)]+tensor.shape.map(Double.init)+tensor.values
                }
                guard f.expected["targetContract"] == LearningTensor(shape:[format.count],values:format) else {throw LearningContractError("Loss target-contract gold must describe the actual ordered supplied roles, types and values exactly.")}
            }
        }
        let attention = AttentionLearningTask.allCases.first { id == "assessment."+$0.rawValue }
        let criterionLimit = attention.map { $0.outputRoles.count+$0.sourceRoles.count+$0.parameterShapes.count+2 } ?? 32
        if let task=attention {
            guard objectiveID == "attention."+task.rawValue,contentVersion == ContractVersion(),assessmentVersion == ContractVersion(),runtime.backend == "pytorch",runtime.capabilities == [task.componentID:ContractVersion(),"attention_mask":ContractVersion()] else {throw LearningContractError("Attention fixture identity and capabilities must match the reviewed lesson.")}
            let expectedKeys=Set(Array(task.checkpointSampleShapes.keys)+task.outputRoles+task.sourceRoles.map {$0+"Gradient"}+task.parameterShapes.keys.map {"parameter."+$0+"Gradient"})
            let inputKeys=Set(task.sourceRoles+task.outputRoles.map {"upstream."+$0}+task.parameterShapes.keys.map {"parameter."+$0})
            guard Set(criteria.map(\.id)) == expectedKeys else {throw LearningContractError("Attention requires every result and influence criterion.")}
            for criterion in criteria {
                guard case .tensor(let key,let absolute,let relative)=criterion.rule,key == criterion.id,absolute.isFinite,relative.isFinite,(0...3e-6).contains(absolute),(0...3e-5).contains(relative) else {throw LearningContractError("Attention criteria require bounded independent numerical checks.")}
                if task.checkpointSampleShapes[key] != nil {guard absolute == 0,relative == 0 else {throw LearningContractError("Blocked-position checks require exact whole-number tables.")}}
            }
            for f in fixtures {
                guard Set(f.inputs.keys)==inputKeys,Set(f.expected.keys)==expectedKeys,let n=f.inputs[task.sourceRoles[0]]?.shape.first,(1...3).contains(n) else {throw LearningContractError("Attention fixtures require exact supplied roles and bounded observation groups.")}
                let checkpoints=task.expectedCheckpoints(observations:n)
                for (key,target) in checkpoints {guard f.expected[key] == target else {throw LearningContractError("Attention checkpoint gold requires the exact declared binary positions and observation geometry.")}}
                for role in task.sourceRoles {guard f.inputs[role]?.shape == [n]+task.sampleShapes[role]!,f.expected[role+"Gradient"]?.shape == [n]+task.sampleShapes[role]! else {throw LearningContractError("Attention source geometry differs from the lesson.")}}
                for role in task.outputRoles {guard f.inputs["upstream."+role]?.shape == [n]+task.outputSampleShapes[role]!,f.expected[role]?.shape == [n]+task.outputSampleShapes[role]! else {throw LearningContractError("Attention result geometry differs from the lesson.")}}
                for (name,shape) in task.parameterShapes {guard f.inputs["parameter."+name]?.shape == shape,f.expected["parameter."+name+"Gradient"]?.shape == shape else {throw LearningContractError("Attention learned-table geometry differs from the lesson.")}}
            }
        }
        let versions=[contentVersion,assessmentVersion]+Array(runtime.capabilities.values)
        guard versions.allSatisfy({[$0.major,$0.minor,$0.patch].allSatisfy({(0...Int(Int32.max)).contains($0)})}) else {throw LearningContractError("Assessment and capability versions must be nonnegative bounded integers.")}
        guard !id.isEmpty,!objectiveID.isEmpty,!runtime.backend.isEmpty,!runtime.version.isEmpty,!runtime.capabilities.isEmpty,(1...32).contains(fixtures.count),(1...criterionLimit).contains(criteria.count),Set(fixtures.map(\.id)).count==fixtures.count,Set(criteria.map(\.id)).count==criteria.count,budget.milliseconds.isFinite,budget.milliseconds>0,budget.milliseconds<=60_000,(1...1_000_000).contains(budget.maximumTensorElements) else {throw LearningContractError("Invalid assessment identity, fixture/criterion bounds or compute budget.")}
        for fixture in fixtures {guard !fixture.id.isEmpty,fixture.inputs.count <= (attention.map {$0.sourceRoles.count+$0.outputRoles.count+$0.parameterShapes.count} ?? 16),!fixture.expected.isEmpty,fixture.expected.count <= (attention.map {$0.outputRoles.count+$0.sourceRoles.count+$0.parameterShapes.count+2} ?? 16),Array(fixture.inputs.values).allSatisfy(\.wellFormed),Array(fixture.expected.values).allSatisfy(\.wellFormed) else {throw LearningContractError("Invalid private fixture tensor.")}}
        for criterion in criteria {guard !criterion.id.isEmpty,!criterion.description.isEmpty,!criterion.nextStep.isEmpty else {throw LearningContractError("Each criterion needs identity, explanation and feedback.")};if case .tensor(let key,let absolute,let relative)=criterion.rule {guard absolute.isFinite,relative.isFinite,(0...1).contains(absolute),(0...1).contains(relative),fixtures.allSatisfy({$0.expected[key] != nil}) else {throw LearningContractError("Invalid numerical tolerance or missing expected output.")}}}
    }
}
public struct LearningFixtureEvidence:Codable,Equatable {
    public var fixtureID:String;public var tensors:[String:LearningTensor];public var partitions:[String:[String]]
    public var composition:ModuleCompositionLearningMetadata?
    public init(fixtureID:String,tensors:[String:LearningTensor],partitions:[String:[String]]=[:],composition:ModuleCompositionLearningMetadata?=nil){self.fixtureID=fixtureID;self.tensors=tensors;self.partitions=partitions;self.composition=composition}
}
/// Trusted runner-adapter output, not a user-authored proof. The adapter must bind measured results to the submission/fixtures.
public struct LearningEvidence:Codable,Equatable {
    public var assessmentID:String;public var assessmentVersion:ContractVersion;public var fixtureDigest:String;public var runtime:LearningRuntime;public var submissionDigest:String
    public var fixtures:[LearningFixtureEvidence];public var elapsedMilliseconds:Double;public var maximumTensorElements:Int;public var cancelled:Bool;public var backendError:String?
    public init(assessmentID:String,assessmentVersion:ContractVersion,fixtureDigest:String,runtime:LearningRuntime,submissionDigest:String,fixtures:[LearningFixtureEvidence],elapsedMilliseconds:Double,maximumTensorElements:Int,cancelled:Bool=false,backendError:String?=nil){self.assessmentID=assessmentID;self.assessmentVersion=assessmentVersion;self.fixtureDigest=fixtureDigest;self.runtime=runtime;self.submissionDigest=submissionDigest;self.fixtures=fixtures;self.elapsedMilliseconds=elapsedMilliseconds;self.maximumTensorElements=maximumTensorElements;self.cancelled=cancelled;self.backendError=backendError}
}
public enum LearningGradeStatus:String,Codable {case passed,failed,incomplete,unavailable,cancelled,executionFailed,budgetExceeded}
public struct LearningCriterionOutcome:Codable,Equatable {
    public var criterionID:String;public var passed:Bool;public var evidence:String;public var nextStep:String
    public init(criterionID:String,passed:Bool,evidence:String,nextStep:String){self.criterionID=criterionID;self.passed=passed;self.evidence=evidence;self.nextStep=nextStep}
}
public struct LearningGrade:Codable,Equatable {
    public var status:LearningGradeStatus;public var summary:String;public var outcomes:[LearningCriterionOutcome]
    public var assessmentID:String?;public var objectiveID:String?;public var contentVersion:ContractVersion?;public var assessmentVersion:ContractVersion?;public var fixtureDigest:String?
    public init(status:LearningGradeStatus,summary:String,outcomes:[LearningCriterionOutcome]=[],assessmentID:String?=nil,objectiveID:String?=nil,contentVersion:ContractVersion?=nil,assessmentVersion:ContractVersion?=nil,fixtureDigest:String?=nil){self.status=status;self.summary=summary;self.outcomes=outcomes;self.assessmentID=assessmentID;self.objectiveID=objectiveID;self.contentVersion=contentVersion;self.assessmentVersion=assessmentVersion;self.fixtureDigest=fixtureDigest}
}
public struct LearningContractError:Error,LocalizedError {public var message:String;public init(_ message:String){self.message=message};public var errorDescription:String?{message}}
func learningHash(_ data:Data)->String {SHA256.hash(data:data).map{String(format:"%02x",$0)}.joined()}
public enum LearningCurriculum {
    public static func validate(_ objectives:[LearningObjective])throws {
        let ids=Set(objectives.map(\.id));guard objectives.count<=256,ids.count==objectives.count,!ids.contains("") else {throw LearningContractError("Objective identities must be unique and bounded.")}
        for objective in objectives {guard Set(objective.prerequisiteIDs).isSubset(of:ids),Set(objective.prerequisiteIDs).count==objective.prerequisiteIDs.count else {throw LearningContractError("Unknown or duplicate prerequisite.")}}
        var complete=Set<String>();while complete.count<objectives.count {let ready=objectives.filter{!complete.contains($0.id) && Set($0.prerequisiteIDs).isSubset(of:complete)};guard !ready.isEmpty else {throw LearningContractError("Objective prerequisites contain a cycle.")};complete.formUnion(ready.map(\.id))}
    }
}

/// Safe learner-facing projection. Hidden fixtures, seeds and expected tensors are intentionally absent.
public struct LearningPublicTask:Codable,Equatable {
    public var assessmentID:String;public var objectiveID:String;public var contentVersion:ContractVersion;public var assessmentVersion:ContractVersion
    public var mode:LearningMode;public var instructions:String;public var criteria:[LearningCriterion];public var budget:LearningBudget;public var runtime:LearningRuntime
    public var prompts:[String]
}
public extension LearningAssessmentPlan {
    func publicTask(mode:LearningMode)->LearningPublicTask {
        let prompts:[String]
        switch mode {
        case .guided:prompts=["Predict the dimensions and values before building.","Construct the graph, then inspect actual measured outputs.","Explain any difference between your prediction and the measurements."]
        case .practice:prompts=["Choose a valid construction and justify its operations.","Use the stated criteria to inspect your result before requesting assessment."]
        case .assessment:prompts=["Submit the graph for independent fixture execution within the declared budget.","Equivalent outputs are accepted; names and layout do not determine correctness."]
        }
        return LearningPublicTask(assessmentID:id,objectiveID:objectiveID,contentVersion:contentVersion,assessmentVersion:assessmentVersion,mode:mode,instructions:task,criteria:criteria,budget:budget,runtime:runtime,prompts:prompts)
    }
}
