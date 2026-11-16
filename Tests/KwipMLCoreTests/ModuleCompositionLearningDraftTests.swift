import Foundation
import XCTest
@testable import KwipMLCore

final class ModuleCompositionLearningDraftTests:XCTestCase {
    private let projectID="DA0A0667-64F2-487E-9C0A-91F58CB7AFDA"
    private func model(_ task:ModuleCompositionLearningTask,prepared:Bool=true)throws->ModelAuthoringDocument {
        let source = try prepared ? ModuleCompositionLearningExamples.preparedSource(task:task):ModuleCompositionLearningExamples.defaultSource(task:task)
        return .init(projectID:projectID,source:source,schemaVersion:2)
    }
    func testRegisteredDraftUsesOneProtectedSourceAndColdIdentity()throws {
        for task in ModuleCompositionLearningTask.allCases {
            XCTAssertTrue(TensorLearningTask.operations.contains(task.rawValue))
            XCTAssertEqual(TensorLearningTask.title(for:task.rawValue),task.title)
            XCTAssertEqual(try TensorLearningTask.plan(operation:task.rawValue,runtimeVersion:"2.14.1",seed:42),try task.assessment(seed:42,runtimeVersion:"2.14.1"))
            let draft=TensorLearningDraft(projectID:projectID,operation:task.rawValue,mode:.practice)
            let source=try XCTUnwrap(draft.compositionModel)
            XCTAssertEqual(source,try ModuleCompositionLearningDraft.model(projectID:projectID,task:task))
            XCTAssertTrue(draft.graph.nodes.isEmpty);XCTAssertTrue(draft.graph.edges.isEmpty)
            XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()),draft)
            XCTAssertThrowsError(try draft.submissionData())
        }
    }
    func testPendingBlockSurvivesColdDraftAndCannotSubmit()throws {
        let task=ModuleCompositionLearningTask.buildReuse,registry=try ComponentRegistry.builtIn()
        var draft=TensorLearningDraft(projectID:projectID,operation:task.rawValue,mode:.guided)
        draft.compositionModel=try ModelBlockEditing.beginDraft(model(task,prepared:false),path:["block","left"])
        try ModuleCompositionLearningDraft.validate(model:draft.compositionModel!,task:task,projectID:projectID,complete:false)
        XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()),draft)
        XCTAssertThrowsError(try draft.submissionData(compositionRegistry:registry))
        var changed=draft.compositionModel!
        changed.drafts[0].proposed.body.nodes[0].config["out_features"]=2
        XCTAssertThrowsError(try ModuleCompositionLearningDraft.validate(model:changed,task:task,projectID:projectID,complete:false))
        changed=draft.compositionModel!;changed.drafts[0].baseRevision+=1
        XCTAssertThrowsError(try ModuleCompositionLearningDraft.validate(model:changed,task:task,projectID:projectID,complete:false))
    }
    func testCompleteCopyOnWriteKeepsPrivateSiblingAndAcceptsFreshReferences()throws {
        let task=ModuleCompositionLearningTask.buildReuse,registry=try ComponentRegistry.builtIn()
        let original=try model(task),pending=try ModelBlockEditing.beginDraft(original,path:["block","left"])
        var counter=0
        let applied=try ModelBlockEditing.applyDraft(pending,path:["block","left"],expectedBaseRevision:0,registry:registry,idProvider:{counter+=1;return "private-copy-\(counter)"})
        XCTAssertEqual(applied.source.definitions.count,original.source.definitions.count+2)
        XCTAssertEqual(Array(applied.source.definitions.prefix(original.source.definitions.count)),original.source.definitions)
        let outer=try XCTUnwrap(applied.source.definitions.first {$0.reference == applied.source.root.instances[0].definition})
        XCTAssertNotEqual(outer.body.instances.first {$0.id == "left"}?.definition,outer.body.instances.first {$0.id == "right"}?.definition)
        try ModuleCompositionLearningDraft.validate(model:applied,task:task,projectID:projectID,complete:true,registry:registry)
        let submission=try ModuleCompositionLearningDraft.submission(model:applied,task:task,registry:registry)
        XCTAssertEqual(try ModuleCompositionLearningSubmission.decode(submission.encoded(),task:task,registry:registry),submission)
    }
    func testPartialLessonApplyDoesNotRelaxOrdinaryApply()throws {
        let task=ModuleCompositionLearningTask.buildReuse,registry=try ComponentRegistry.builtIn()
        let pending=try ModelBlockEditing.beginDraft(model(task,prepared:false),path:["block","left"])
        XCTAssertThrowsError(try ModelBlockEditing.applyDraft(pending,path:["block","left"],expectedBaseRevision:0,registry:registry))
        var counter=0
        let applied=try ModelBlockEditing.applyDraft(pending,path:["block","left"],expectedBaseRevision:0,registry:registry,definitionValidator:ModuleCompositionLearningDraft.definitionValidator(task:task),idProvider:{counter+=1;return "partial-copy-\(counter)"})
        XCTAssertTrue(applied.drafts.isEmpty)
        try ModuleCompositionLearningDraft.validate(model:applied,task:task,projectID:projectID,complete:false)
        XCTAssertThrowsError(try ModuleCompositionLearningDraft.submission(model:applied,task:task,registry:registry))
        var forged=pending;forged.drafts[0].proposed.body.nodes[0].config["bias"]=false
        XCTAssertThrowsError(try ModelBlockEditing.applyDraft(forged,path:["block","left"],expectedBaseRevision:0,registry:registry,definitionValidator:ModuleCompositionLearningDraft.definitionValidator(task:task)))
    }
    func testProtectedMembershipForeignRetiredDefinitionsAndWrongProjectRefuse()throws {
        let task=ModuleCompositionLearningTask.orderName
        var value=try model(task)
        try ModuleCompositionLearningDraft.validate(model:value,task:task,projectID:projectID,complete:false)
        value.source.definitions[0].body.nodes.removeLast()
        XCTAssertThrowsError(try ModuleCompositionLearningDraft.validate(model:value,task:task,projectID:projectID,complete:false))
        value=try model(task);var foreign=value.source.definitions[0];foreign.reference = .init("foreign-retired");foreign.body.nodes[0].type="relu";value.source.definitions.append(foreign)
        XCTAssertThrowsError(try ModuleCompositionLearningDraft.validate(model:value,task:task,projectID:projectID,complete:false))
        value=try model(task)
        XCTAssertThrowsError(try ModuleCompositionLearningDraft.validate(model:value,task:task,projectID:UUID().uuidString,complete:false))
    }
    func testExplicitRegistryRequiredAndMissingMeasuredMetadataCannotReplay()throws {
        let task=ModuleCompositionLearningTask.orderName,registry=try ComponentRegistry.builtIn()
        var draft=TensorLearningDraft(projectID:projectID,operation:task.rawValue,mode:.guided)
        draft.compositionModel=try model(task)
        XCTAssertThrowsError(try draft.submissionData())
        let submission=try draft.submissionData(compositionRegistry:registry)
        let plan=try task.assessment(seed:42,runtimeVersion:"2.14.1")
        // Admission-only synthetic transport, never a native execution proof.
        let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(submission),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:128)
        XCTAssertThrowsError(try TensorLearningReceipt(projectID:projectID,operation:task.rawValue,mode:.guided,submissionData:submission,plan:plan,evidence:evidence))
        XCTAssertThrowsError(try TensorLearningReceipt(projectID:projectID,operation:task.rawValue,mode:.guided,submissionData:submission,plan:plan,evidence:evidence,compositionRegistry:registry))
    }
    func testLegacyDraftOmitsCompositionFieldsAndRejectsForeignSource()throws {
        var draft=TensorLearningDraft(projectID:projectID,operation:"reshape",mode:.guided)
        let object=try XCTUnwrap(JSONSerialization.jsonObject(with:draft.encoded()) as? [String:Any])
        XCTAssertNil(object["compositionModel"])
        draft.compositionModel=try model(.buildReuse)
        XCTAssertThrowsError(try draft.validate())
    }
    func testRecordedRegistryReceiptRoundTripRefusesStaleUnknownAndMissingNativeRecord()throws {
        let task=ModuleCompositionLearningTask.orderName,registry=try ComponentRegistry.builtIn()
        let source=try ModuleCompositionLearningExamples.preparedSource(task:task)
        let expansion=try CompositeExpander.expand(source,registry:registry,context:.explicitObjective)
        let nativePlan=try XCTUnwrap(expansion.graph.compositionPlan)
        var containers:[ModuleCompositionLearningMetadata.Container]=[],owners:[ModuleCompositionLearningMetadata.Owner]=[]
        func visit(_ container:CompositionPlan.Container,_ path:String)throws {
            containers.append(.init(path:path,kind:container.kind,keys:container.children.map(\.key),matches:true))
            for child in container.children {
                let name=path+(container.kind == .graph ? ".blocks.":".")+child.key
                if child.kind == .container {try visit(XCTUnwrap(nativePlan.containers.first {$0.id==child.id}),name)}
                else if let node=expansion.graph.nodes.first(where:{$0.id==child.id}),node.type=="linear" {
                    owners.append(.init(node:node.id,parameter:"weight",name:name+".operation.weight",shape:[1,1],requiresGrad:true))
                    owners.append(.init(node:node.id,parameter:"bias",name:name+".operation.bias",shape:[1],requiresGrad:true))
                }
            }
        }
        try visit(XCTUnwrap(nativePlan.containers.first),"composition_root")
        let order=GraphValidator.validate(expansion.graph,registry:registry,context:.explicitObjective).orderedNodeIDs
        let metadata=ModuleCompositionLearningMetadata(nativeOwners:owners,nativeContainers:containers,invocations:order.flatMap{[.init(phase:"started",node:$0),.init(phase:"completed",node:$0)]},savedStateKeys:owners.map(\.name),independentStorage:true)
        let submission=try ModuleCompositionLearningSubmission(source:source,task:task,registry:registry).encoded(),plan=try task.assessment(seed:42,runtimeVersion:"2.14.1")
        // Synthetic receipt admission fixture; actual native run proof is a separate App/runtime test.
        let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(submission),fixtures:plan.fixtures.map{.init(fixtureID:$0.id,tensors:$0.expected,composition:metadata)},elapsedMilliseconds:1,maximumTensorElements:128)
        let receipt=try TensorLearningReceipt(projectID:projectID,operation:task.rawValue,mode:.guided,submissionData:submission,plan:plan,evidence:evidence,compositionRegistry:registry)
        XCTAssertEqual(try TensorLearningReceipt.decode(receipt.encoded()).compositionRegistry,registry)
        var bad=receipt;bad.compositionRegistrySHA256=String(repeating:"0",count:64)
        XCTAssertThrowsError(try bad.verifiedReplay(expectedProjectID:projectID))
        bad=receipt;bad.evidence.fixtures[0].composition=nil
        XCTAssertThrowsError(try bad.verifiedReplay(expectedProjectID:projectID))
        var object=try XCTUnwrap(JSONSerialization.jsonObject(with:receipt.encoded()) as? [String:Any])
        var rawRegistry=object["compositionRegistry"] as! [String:Any]
        var components=rawRegistry["components"] as! [[String:Any]]
        components[0]["unverifiedPortAuthority"]=true;rawRegistry["components"]=components;object["compositionRegistry"]=rawRegistry
        XCTAssertThrowsError(try TensorLearningReceipt.decode(JSONSerialization.data(withJSONObject:object)))
        bad=receipt;bad.operation="reshape"
        XCTAssertThrowsError(try bad.verifiedReplay(expectedProjectID:projectID))
    }
    func testCompositionDraftRejectsUnknownNestedFieldsNullAndExtraFlatGraph()throws {
        let task=ModuleCompositionLearningTask.orderName
        var draft=TensorLearningDraft(projectID:projectID,operation:task.rawValue,mode:.guided)
        draft.compositionModel=try model(task)
        let base=try XCTUnwrap(JSONSerialization.jsonObject(with:draft.encoded()) as? [String:Any])
        var object=base;object["compositionModel"]=NSNull()
        XCTAssertThrowsError(try TensorLearningDraft.decode(JSONSerialization.data(withJSONObject:object)))
        object=base;var rawModel=object["compositionModel"] as! [String:Any];rawModel["answers"]=[0];object["compositionModel"]=rawModel
        XCTAssertThrowsError(try TensorLearningDraft.decode(JSONSerialization.data(withJSONObject:object)))
        draft.graph.compositionPlan=try CompositeExpander.expand(draft.compositionModel!.source,registry:ComponentRegistry.builtIn(),context:.explicitObjective).graph.compositionPlan
        XCTAssertThrowsError(try draft.validate())
    }

    func testInvalidPublicProjectIdentityConstructsWithoutFatalAndRefusesValidation() {
        for task in ModuleCompositionLearningTask.allCases {
            let draft=TensorLearningDraft(projectID:"not-a-project-uuid",operation:task.rawValue,mode:.guided)
            XCTAssertNil(draft.compositionModel)
            XCTAssertThrowsError(try draft.validate())
        }
    }

}
