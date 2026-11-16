import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class LossWorkflowLifecycleTests:XCTestCase {
    private let run=UUID().uuidString,pin=String(repeating:"a",count:64)
    private func value(_ mode:WorkflowTrainingMode,objective:Bool,parentObjective:Bool=true,scheduler:Bool=false)->[String:JSONValue] {
        let base=["parameters","optimizer","randomGenerator","epoch","metrics"]
        let schedule=scheduler ? ["scheduler","rateTrace"]:[]
        let selected=objective ? ["objective","objectiveBuffers"]:[]
        let all=base+(mode == .fresh ? schedule+selected:selected+schedule)
        var lineage:[String:JSONValue]=["mode":.string(mode == .fresh ? "newTraining":mode == .exactResume ? "exactResume":"weightsOnlyFineTune"),"parentRunID":mode == .fresh ? .null:.string(run),"restoredState":.array((mode == .fresh ? []:mode == .exactResume ? all:["parameters"]).map(JSONValue.string)),"resetState":.array((mode == .fresh ? all:mode == .exactResume ? []:Array(base.dropFirst())+(parentObjective ? ["objective","objectiveBuffers"]:[])+schedule+((objective && !parentObjective) ? ["objective","objectiveBuffers"]:[])).map(JSONValue.string))]
        if mode != .fresh {lineage["parentCheckpointSHA256"] = .string(pin);lineage["parentCompletedEpoch"] = 2;lineage["guarantee"] = .string(mode == .exactResume ? "epochBoundaryExactCPU":"weightsOnly")}
        return ["lineage":.object(lineage),"settings":.object(["initialEpoch":mode == .exactResume ? 2:0,"epochsExecuted":mode == .exactResume ? 1:3]),"runtime":.object(["device":.string("cpu")])]
    }
    func testAllEightObjectiveFreshExactAndWeightsOnlyStateLists()throws {
        for _ in LossConfiguration.componentIDs {for mode in WorkflowTrainingMode.allCases {for scheduler in [false,true] {
            let parent:WorkflowLifecycleParentEvidence?=mode == .fresh ? nil:.init(workerRunID:run,checkpointSHA256:pin,completedEpoch:2,hasScheduler:scheduler,hasObjective:true)
            let manifest=value(mode,objective:true,scheduler:scheduler)
            XCTAssertNoThrow(try WorkflowLifecycleVerifier.validate(manifest,mode:mode,targetEpochs:3,parent:parent,hasScheduler:scheduler,hasObjective:true))
            var wrong=manifest
            if case .object(var lineage)=wrong["lineage"] {let key=mode == .exactResume ? "restoredState":"resetState";lineage[key] = .array([.string("objective")]);wrong["lineage"] = .object(lineage)}
            XCTAssertThrowsError(try WorkflowLifecycleVerifier.validate(wrong,mode:mode,targetEpochs:3,parent:parent,hasScheduler:scheduler,hasObjective:true))
        }}}
    }
    func testExactResumeRequiresObjectivePresenceParityAndHonestCPU()throws {
        let parent=WorkflowLifecycleParentEvidence(workerRunID:run,checkpointSHA256:pin,completedEpoch:2,hasObjective:false)
        XCTAssertThrowsError(try WorkflowLifecycleVerifier.validate(value(.exactResume,objective:true),mode:.exactResume,targetEpochs:3,parent:parent,hasObjective:true))
        var mps=value(.exactResume,objective:true);mps["runtime"] = .object(["device":.string("mps")])
        XCTAssertThrowsError(try WorkflowLifecycleVerifier.validate(mps,mode:.exactResume,targetEpochs:3,parent:.init(workerRunID:run,checkpointSHA256:pin,completedEpoch:2,hasObjective:true),hasObjective:true))
    }
    func testWeightsOnlyResetsObjectiveWhenOldOrNewSideOwnsIt()throws {
        for old in [false,true] {for selected in [false,true] {
            let parent=WorkflowLifecycleParentEvidence(workerRunID:run,checkpointSHA256:pin,completedEpoch:2,hasObjective:old)
            XCTAssertNoThrow(try WorkflowLifecycleVerifier.validate(value(.fineTune,objective:selected,parentObjective:old),mode:.fineTune,targetEpochs:3,parent:parent,hasObjective:selected))
        }}
    }
}
