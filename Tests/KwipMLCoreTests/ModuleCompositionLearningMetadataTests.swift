import Foundation
import XCTest
@testable import KwipMLCore

final class ModuleCompositionLearningMetadataTests:XCTestCase {
    private let encoder:JSONEncoder = {let e=JSONEncoder();e.outputFormatting=[.sortedKeys,.withoutEscapingSlashes];return e}()
    // Synthetic transport fixture constructed from independently expanded source, not a native receipt.
    private func fixture(_ task:ModuleCompositionLearningTask, source supplied:CompositeProjectDocument?=nil)throws->(CompositeProjectDocument,ModuleCompositionLearningMetadata) {
        let source=try supplied ?? ModuleCompositionLearningExamples.preparedSource(task:task)
        let registry=try ComponentRegistry.builtIn()
        let expansion=try CompositeExpander.expand(source,registry:registry,context:.explicitObjective)
        let plan=try XCTUnwrap(expansion.graph.compositionPlan)
        var containers:[ModuleCompositionLearningMetadata.Container]=[],owners:[ModuleCompositionLearningMetadata.Owner]=[]
        func visit(_ container:CompositionPlan.Container,_ path:String)throws {
            containers.append(.init(path:path,kind:container.kind,keys:container.children.map(\.key),matches:true))
            for child in container.children {
                let name=path+(container.kind == .graph ? ".blocks.":".")+child.key
                if child.kind == .container {try visit(XCTUnwrap(plan.containers.first {$0.id == child.id}),name)}
                else if let node=expansion.graph.nodes.first(where:{$0.id == child.id}),node.type == "linear" {
                    let origin=try XCTUnwrap(expansion.nodeOrigins[node.id])
                    let input=try XCTUnwrap(node.config["in_features"]?.integerValue),output=try XCTUnwrap(node.config["out_features"]?.integerValue)
                    for (parameter,shape) in [("weight",[output,input]),("bias",[output])] {
                        let fixed=["negate","difference"].contains(origin.localNodeID) || (task == .buildReuse && origin.instancePath == ["block","left"] && origin.localNodeID == "a" && parameter == "weight")
                        owners.append(.init(node:node.id,parameter:parameter,name:name+".operation."+parameter,shape:shape,requiresGrad:!fixed))
                    }
                }
            }
        }
        try visit(XCTUnwrap(plan.containers.first),"composition_root")
        let order=GraphValidator.validate(expansion.graph,registry:registry,context:.explicitObjective).orderedNodeIDs
        let invocations=order.flatMap { node in [ModuleCompositionLearningMetadata.Invocation(phase:"started",node:node),.init(phase:"completed",node:node)] }
        return (source,.init(nativeOwners:owners,nativeContainers:containers,invocations:invocations,savedStateKeys:owners.map(\.name),independentStorage:true))
    }
    func testCurrentNestedSourceMetadataColdRoundTripAndAfterChecksFlags()throws {
        let registry=try ComponentRegistry.builtIn()
        for task in ModuleCompositionLearningTask.allCases {
            let (source,metadata)=try fixture(task)
            let bytes=try encoder.encode(metadata),decoded=try ModuleCompositionLearningMetadata.decode(bytes)
            XCTAssertEqual(decoded,metadata)
            XCTAssertNoThrow(try decoded.validate(source:source,task:task,registry:registry))
            XCTAssertFalse(String(decoding:bytes,as:UTF8.self).contains("sourceDigest"))
            XCTAssertFalse(String(decoding:bytes,as:UTF8.self).contains("coefficients"))
            if task == .orderName {
                XCTAssertTrue(metadata.nativeOwners.allSatisfy(\.requiresGrad))
                XCTAssertTrue(metadata.nativeOwners.contains {$0.name.contains(".finish.operation.weight")})
                XCTAssertTrue(metadata.nativeOwners.contains {$0.name.contains(".start.operation.bias")})
            } else {XCTAssertEqual(metadata.nativeOwners.filter {!$0.requiresGrad}.count,5)}
        }
    }
    func testLegacyFixtureOmitsOptionalMetadataAndRetainsExactBytes()throws {
        let old=LearningFixtureEvidence(fixtureID:"legacy",tensors:[:])
        XCTAssertEqual(String(decoding:try encoder.encode(old),as:UTF8.self),#"{"fixtureID":"legacy","partitions":{},"tensors":{}}"#)
        XCTAssertNil(try JSONDecoder().decode(LearningFixtureEvidence.self,from:encoder.encode(old)).composition)
        let (_,metadata)=try fixture(.orderName)
        let selected=LearningFixtureEvidence(fixtureID:"current",tensors:[:],composition:metadata)
        XCTAssertEqual(try JSONDecoder().decode(LearningFixtureEvidence.self,from:encoder.encode(selected)),selected)
    }
    func testUnknownNestedFieldsMalformedTypesAndBudgetsRefuse()throws {
        let (_,metadata)=try fixture(.orderName)
        let base=try XCTUnwrap(JSONSerialization.jsonObject(with:encoder.encode(metadata)) as? [String:Any])
        for mutation in ["unknown","owner","container","invocation","version","null","limit"] {
            var object=base
            switch mutation {
            case "unknown":object["sourceDigest"]="unqualified"
            case "owner":var owners=object["nativeOwners"] as! [[String:Any]];owners[0]["rawValues"]=[1];object["nativeOwners"]=owners
            case "container":var containers=object["nativeContainers"] as! [[String:Any]];containers[0]["matches"]=1;object["nativeContainers"]=containers
            case "invocation":object["invocations"]=[["started","input","extra"]]
            case "version":object["schemaVersion"]=true
            case "null":object["savedStateKeys"]=NSNull()
            default:object["nativeOwners"]=Array(repeating:(object["nativeOwners"] as! [[String:Any]])[0],count:257)
            }
            XCTAssertThrowsError(try ModuleCompositionLearningMetadata.decode(JSONSerialization.data(withJSONObject:object)),mutation)
        }
        XCTAssertThrowsError(try ModuleCompositionLearningMetadata.decode(Data(repeating:32,count:ModuleCompositionLearningMetadata.maximumBytes+1)))
    }
    func testSourceBoundOwnersPathsOrderStatesAndInvocationDependencies()throws {
        let task=ModuleCompositionLearningTask.orderName,registry=try ComponentRegistry.builtIn()
        let (source,good)=try fixture(task)
        for change in ["path","shape","order","flag","state","container","phase","node","dependency"] {
            var bad=good
            switch change {
            case "path":bad.nativeOwners[0].name="layers.forged.weight"
            case "shape":bad.nativeOwners[0].shape=[2,2]
            case "order":bad.nativeOwners.swapAt(0,1)
            case "flag":bad.nativeOwners[0].requiresGrad=false
            case "state":bad.savedStateKeys.reverse()
            case "container":bad.nativeContainers[1].keys.reverse()
            case "phase":bad.invocations[0].phase="backward"
            case "node":bad.invocations[0].node="foreign"
            default:bad.invocations.swapAt(0,2);bad.invocations.swapAt(1,3)
            }
            XCTAssertThrowsError(try bad.validate(source:source,task:task,registry:registry),change)
        }
        var measuredFalse=good;measuredFalse.independentStorage=false;measuredFalse.nativeContainers[0].matches=false
        XCTAssertNoThrow(try measuredFalse.validate(source:source,task:task,registry:registry),"Measured false is evidence of failure, not forged success")
        var renamed=source
        let named=try XCTUnwrap(renamed.definitions.firstIndex {$0.reference.id == "named"})
        let policy=try XCTUnwrap(renamed.definitions[named].composition)
        var children=policy.children
        children[0] = .init(localID:children[0].localID,key:"renamedFinish")
        renamed.definitions[named].composition = .init(kind:policy.kind,children:children)
        let (_,current)=try fixture(task,source:renamed)
        XCTAssertNoThrow(try current.validate(source:renamed,task:task,registry:registry),"Compare submitted construction, not gold")
        XCTAssertThrowsError(try good.validate(source:renamed,task:task,registry:registry),"Stale paths from previous source")
    }
    func testProjectionRequiresCurrentPublicModeAndCompleteMetadata()throws {
        let task=ModuleCompositionLearningTask.orderName,registry=try ComponentRegistry.builtIn()
        let (source,metadata)=try fixture(task)
        let submission=try ModuleCompositionLearningSubmission(source:source,task:task,registry:registry).encoded()
        let plan=try task.assessment(seed:42,runtimeVersion:"2.14.1")
        var evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(submission),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected,composition:metadata)},elapsedMilliseconds:1,maximumTensorElements:128)
        func project(_ mode:LearningMode = .guided,_ attempt:LearningMode = .guided)->[ModuleCompositionLearningMetadata]? {ModuleCompositionLearningMetadata.project(plan:plan,evidence:evidence,currentSubmission:submission,registry:registry,mode:mode,attemptMode:attempt)}
        XCTAssertEqual(project()?.count,4)
        XCTAssertEqual(project(.practice,.practice)?.count,4)
        XCTAssertNil(project(.assessment,.assessment));XCTAssertNil(project(.practice,.guided))
        evidence.submissionDigest=String(repeating:"0",count:64);XCTAssertNil(project())
        evidence.submissionDigest=learningHash(submission);evidence.fixtures[0].composition=nil;XCTAssertNil(project())
    }
}
