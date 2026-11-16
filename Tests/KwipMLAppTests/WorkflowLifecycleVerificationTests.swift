import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowLifecycleVerificationTests:XCTestCase {
    private let run=UUID().uuidString
    private let pin=String(repeating:"a",count:64)
    private func manifest(_ mode:WorkflowTrainingMode)->[String:JSONValue] {
        let all=["parameters","optimizer","randomGenerator","epoch","metrics"]
        var lineage:[String:JSONValue]=[
            "mode":.string(mode == .fresh ? "newTraining" : mode == .exactResume ? "exactResume" : "weightsOnlyFineTune"),
            "parentRunID":mode == .fresh ? .null : .string(run),
            "restoredState":.array((mode == .fresh ? [] : mode == .exactResume ? all : ["parameters"]).map(JSONValue.string)),
            "resetState":.array((mode == .fresh ? all : mode == .exactResume ? [] : Array(all.dropFirst())).map(JSONValue.string))]
        if mode != .fresh {
            lineage["parentCheckpointSHA256"] = .string(pin)
            lineage["parentCompletedEpoch"] = .integer(3)
            lineage["guarantee"] = .string(mode == .exactResume ? "epochBoundaryExactCPU":"weightsOnly")
        }
        return ["lineage":.object(lineage),"settings":.object(["initialEpoch":.integer(mode == .exactResume ? 3:0),"epochsExecuted":.integer(mode == .exactResume ? 2:5)])]
    }
    func testRequiresExactModeEpochAndStateTransferEvidence()throws {
        for mode in WorkflowTrainingMode.allCases {
            let parent:WorkflowLifecycleParentEvidence? = mode == .fresh ? nil : .init(workerRunID:run,checkpointSHA256:pin,completedEpoch:3)
            let value=manifest(mode)
            XCTAssertNoThrow(try WorkflowLifecycleVerifier.validate(value,mode:mode,targetEpochs:5,parent:parent))
            for other in WorkflowTrainingMode.allCases where other != mode {
                XCTAssertThrowsError(try WorkflowLifecycleVerifier.validate(value,mode:other,targetEpochs:5,parent:parent))
            }
            var bad=value;bad["settings"] = .object(["initialEpoch":.integer(0),"epochsExecuted":.integer(4)])
            XCTAssertThrowsError(try WorkflowLifecycleVerifier.validate(bad,mode:mode,targetEpochs:5,parent:parent))
            bad=value
            if case .object(var lineage)=bad["lineage"] {lineage["restoredState"] = .array([.string("parameters"),.string("parameters")]);bad["lineage"] = .object(lineage)}
            XCTAssertThrowsError(try WorkflowLifecycleVerifier.validate(bad,mode:mode,targetEpochs:5,parent:parent))
            if mode != .fresh {
                XCTAssertThrowsError(try WorkflowLifecycleVerifier.validate(value,mode:mode,targetEpochs:5,parent:nil))
                XCTAssertThrowsError(try WorkflowLifecycleVerifier.validate(value,mode:mode,targetEpochs:5,parent:.init(workerRunID:UUID().uuidString,checkpointSHA256:pin,completedEpoch:3)))
            }
        }
    }
}
