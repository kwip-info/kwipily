import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class LossTargetMappingInspectorTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    private func fixture(_ task:LossLearningTask)throws->(WorkflowAuthoringStore,WorkflowControlNodeContext,LossTargetMappingInspectorContext) {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
        let document=try WorkflowExamples.preparedLossObjective(task:task),library=ProjectLibrary(directory:directory)
        _=try library.saveSource(.workflow(document))
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repository.appendingPathComponent("runner/dataflow-operators.json")))
        let address=WorkflowNodeAddress(domain:.control,localID:"mapping")
        XCTAssertTrue(owner.select(.init(nodeAddresses:[address]),lease:owner.lease()))
        let control=try WorkflowControlNodeContext(owner:owner,address:address)
        return(owner,control,try LossTargetMappingInspectorContext(context:control))
    }
    func testAllEightGenuineMappingsDiscoverQualifiedObjectiveAndExactContracts()throws {
        for task in LossLearningTask.allCases {
            let(owner,control,editor)=try fixture(task)
            XCTAssertTrue(control.hasObjectiveMapping)
            XCTAssertEqual(editor.componentID,task.componentID)
            let existing=try WorkflowControlContracts.lossTargetMapping(control.configuration,address:control.authority.address)
            XCTAssertEqual(editor.draft.mapping,existing)
            XCTAssertNil(editor.validationMessage)
            XCTAssertFalse(editor.hasChanges);XCTAssertFalse(editor.canApply)
            XCTAssertNotNil(editor.inputContract);XCTAssertNotNil(editor.predictionContract)
            XCTAssertEqual(editor.draft.mapping.inputColumns.count,editor.inputContract?.layout.sample.elementCount)
            XCTAssertEqual(owner.document.schemaVersion,14)
        }
    }
    func testOrderedRowsDraftRepairApplyDiscardAndUndoAreSourceOwned()throws {
        let(owner,control,editor)=try fixture(.l1),before=owner.document
        let id=editor.draft.inputs.rows[0].id
        XCTAssertTrue(editor.update {$0.inputs.setName("",for:id)})
        XCTAssertNotNil(editor.validationMessage);XCTAssertFalse(editor.apply())
        XCTAssertEqual(owner.document,before);XCTAssertTrue(owner.hasPendingInspectorEdits)
        XCTAssertTrue(editor.discard());XCTAssertFalse(owner.hasPendingInspectorEdits)
        XCTAssertTrue(editor.update {$0.inputs.move($0.inputs.rows[1].id,by:-1)})
        XCTAssertTrue(editor.apply())
        let mapping=try WorkflowControlContracts.lossTargetMapping(control.configuration,address:control.authority.address)
        XCTAssertEqual(mapping.inputColumns,["context","signal"])
        XCTAssertFalse(owner.hasPendingInspectorEdits)
        XCTAssertTrue(owner.undo(lease:owner.lease()));XCTAssertFalse(editor.apply());XCTAssertFalse(editor.discard())
        XCTAssertEqual(owner.document.nodes.first {$0.address == control.authority.address}?.payload,before.nodes.first {$0.address == control.authority.address}?.payload)
    }
    func testCTCScalarLengthsAndTripletGroupsHaveNoDummyTarget()throws {
        let(_,_,ctc)=try fixture(.ctc)
        XCTAssertEqual(ctc.draft.roles.map(\.name),["target","input_lengths","target_lengths"])
        XCTAssertEqual(ctc.draft.roles.map(\.dtype),[.int64,.int64,.int64])
        XCTAssertEqual(ctc.draft.roles.map(\.sampleShape),[[2],[],[]])
        XCTAssertTrue(ctc.update {$0.roles[1].sampleShape=[1]})
        XCTAssertNotNil(ctc.validationMessage);XCTAssertFalse(ctc.apply())
        XCTAssertTrue(ctc.discard());XCTAssertNil(ctc.validationMessage)
        let(_,control,triplet)=try fixture(.triplet)
        XCTAssertTrue(triplet.draft.roles.isEmpty);XCTAssertEqual(triplet.draft.groups.count,3)
        XCTAssertEqual(triplet.draft.mapping.tripletSourceGroups.flatMap {$0},triplet.draft.mapping.inputColumns)
        XCTAssertNil(control.configuration["targetColumn"])
        XCTAssertTrue(triplet.update {$0.groups[1].add()})
        XCTAssertNotNil(triplet.validationMessage);XCTAssertFalse(triplet.apply())
        XCTAssertTrue(triplet.discard());XCTAssertNil(triplet.validationMessage)
    }
    func testObjectiveSpecificRoleDTypeAndPredictionShapeRefuseUntilRepaired()throws {
        let(_,_,ce)=try fixture(.crossEntropy)
        XCTAssertEqual(ce.draft.roles[0].dtype,.int64);XCTAssertEqual(ce.draft.roles[0].sampleShape,[])
        XCTAssertTrue(ce.update {$0.roles[0].dtype = .float32})
        XCTAssertNotNil(ce.validationMessage);XCTAssertFalse(ce.apply());XCTAssertTrue(ce.discard())
        let(_,_,bce)=try fixture(.bce)
        XCTAssertEqual(bce.draft.roles.map(\.name),["target","weight"])
        XCTAssertTrue(bce.update {$0.roles[0].sampleShape=[1,2]})
        XCTAssertNotNil(bce.validationMessage);XCTAssertFalse(bce.apply())
        XCTAssertTrue(bce.update {$0.roles[0].sampleShape=[2]});XCTAssertNil(bce.validationMessage)
        XCTAssertTrue(bce.update {$0.roles.removeLast()});XCTAssertNotNil(bce.validationMessage);XCTAssertFalse(bce.apply())
    }
    func testBusyForeignRecipeAndWrongConnectionCannotCommitRetainedDraft()throws {
        let(owner,control,editor)=try fixture(.mse)
        XCTAssertTrue(editor.update {$0.roles[0].columns.setName("replacement",for:$0.roles[0].columns.rows[0].id)})
        owner.setExecutionBusy(true);XCTAssertFalse(editor.apply());XCTAssertFalse(editor.discard())
        owner.setExecutionBusy(false)
        let train=WorkflowExamples.trainAddress
        XCTAssertTrue(owner.select(.init(nodeAddresses:[train]),lease:owner.lease()))
        let trainAuthority=try WorkflowNodeEditingAuthority(owner:owner,address:train)
        var config=trainAuthority.configuration
        let objective=try LossConfiguration.decodeRecipe(XCTUnwrap(config["objective"]))
        var values=objective.values;values["reduction"] = .string("sum")
        config["objective"]=try LossConfiguration(componentID:objective.componentID,values:values).recipeValue
        XCTAssertTrue(trainAuthority.configure(config,expected:trainAuthority.configuration))
        XCTAssertFalse(editor.apply());XCTAssertTrue(editor.hasChanges)
        XCTAssertFalse(editor.update {$0.inputs.add()})
        guard case .trainingData(let retained)=owner.document.nodes.first(where:{$0.address == control.authority.address})?.payload else {return XCTFail("Retained source mapping must remain present.")}
        XCTAssertEqual(try WorkflowControlContracts.lossTargetMapping(retained,address:control.authority.address).roles[0].columns,["desired"])
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
        var broken=try WorkflowExamples.preparedLossObjective(task:.mse)
        let edge=try XCTUnwrap(broken.edges.firstIndex {$0.kind == .trainingData && $0.from.address == .init(domain:.control,localID:"mapping") && $0.to.address == WorkflowExamples.trainAddress})
        broken.edges[edge].from.port="wrong"
        let library=ProjectLibrary(directory:directory);_=try library.saveSource(.workflow(broken))
        let invalidOwner=try WorkflowAuthoringStore(library:library,projectID:broken.projectID,modelRegistry:.builtIn(),dataRegistry:owner.dataRegistry)
        let invalidAddress=WorkflowNodeAddress(domain:.control,localID:"mapping")
        XCTAssertTrue(invalidOwner.select(.init(nodeAddresses:[invalidAddress]),lease:invalidOwner.lease()))
        let invalidControl=try WorkflowControlNodeContext(owner:invalidOwner,address:invalidAddress)
        XCTAssertThrowsError(try LossTargetMappingInspectorContext(context:invalidControl))
    }
    func testConnectedObjectiveConvertsLegacyColumnsWithoutDummyValuesAndDisconnectedDraftRefuses()throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
        var document=try WorkflowExamples.preparedLossObjective(task:.crossEntropy)
        let address=WorkflowNodeAddress(domain:.control,localID:"mapping")
        let index=try XCTUnwrap(document.nodes.firstIndex {$0.address == address})
        document.nodes[index].payload = .trainingData(config:["featureColumns":.array([.string("signal"),.string("context")]),"targetColumn":.string("class")])
        let library=ProjectLibrary(directory:directory);_=try library.saveSource(.workflow(document))
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repository.appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertTrue(owner.select(.init(nodeAddresses:[address]),lease:owner.lease()))
        let control=try WorkflowControlNodeContext(owner:owner,address:address),editor=try LossTargetMappingInspectorContext(context:control)
        XCTAssertTrue(editor.canApply);XCTAssertTrue(editor.hasChanges)
        XCTAssertEqual(editor.draft.mapping.roles[0].columns,["class"]);XCTAssertNil(editor.validationMessage)
        XCTAssertTrue(editor.apply());XCTAssertNil(control.configuration["targetColumn"])
        XCTAssertEqual(try WorkflowControlContracts.lossTargetMapping(control.configuration,address:address).roles[0].dtype,.int64)
        XCTAssertTrue(editor.update {$0.inputs.move($0.inputs.rows[1].id,by:-1)})
        let objectiveEdge=try XCTUnwrap(owner.document.edges.first {$0.kind == .trainingData && $0.from.address == address && $0.to.address == WorkflowExamples.trainAddress})
        XCTAssertNotNil(owner.apply(.delete(.init(edgeIDs:[objectiveEdge.id])),lease:owner.lease()))
        XCTAssertFalse(editor.apply());XCTAssertFalse(editor.discard());XCTAssertTrue(editor.hasChanges)
        XCTAssertFalse(editor.update {$0.inputs.add()})
        XCTAssertEqual(try WorkflowControlContracts.lossTargetMapping({if case .trainingData(let config)=owner.document.nodes[index].payload {return config};return [:]}(),address:address).inputColumns,["signal","context"])
    }

    func testChangedObjectiveModeProjectsRequiredRoleFormatWithoutChangingSourceUntilApply()throws {
        func changed(_ task:LossLearningTask,key:String,value:JSONValue)throws->(WorkflowAuthoringStore,WorkflowControlNodeContext,LossTargetMappingInspectorContext) {
            let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
            var document=try WorkflowExamples.preparedLossObjective(task:task)
            let train=try XCTUnwrap(document.nodes.firstIndex {$0.address == WorkflowExamples.trainAddress})
            guard case .train(var settings)=document.nodes[train].payload else {throw DataFlowContractError("Expected Train.")}
            let objective=try LossConfiguration.decodeRecipe(XCTUnwrap(settings["objective"]))
            var values=objective.values;values[key]=value
            settings["objective"]=try LossConfiguration(componentID:objective.componentID,values:values).recipeValue
            document.nodes[train].payload = .train(config:settings)
            let library=ProjectLibrary(directory:directory);_=try library.saveSource(.workflow(document))
            let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repository.appendingPathComponent("runner/dataflow-operators.json")))
            let address=WorkflowNodeAddress(domain:.control,localID:"mapping")
            XCTAssertTrue(owner.select(.init(nodeAddresses:[address]),lease:owner.lease()))
            let control=try WorkflowControlNodeContext(owner:owner,address:address)
            return(owner,control,try LossTargetMappingInspectorContext(context:control))
        }
        let(owner,control,ce)=try changed(.crossEntropy,key:"target_mode",value:.string("probabilities")),before=owner.document
        XCTAssertEqual(ce.draft.roles[0].dtype,.float32);XCTAssertEqual(ce.draft.roles[0].columns.features,["class"])
        XCTAssertNotNil(ce.roleChangeMessage);XCTAssertTrue(ce.hasChanges);XCTAssertNotNil(ce.validationMessage)
        XCTAssertTrue(ce.discard());XCTAssertEqual(owner.document,before)
        XCTAssertTrue(ce.update {$0.roles[0].sampleShape=[3];$0.roles[0].columns=TrainingMappingDraft(features:["share_a","share_b","share_c"],target:"")})
        XCTAssertNil(ce.validationMessage);XCTAssertTrue(ce.apply())
        XCTAssertEqual(try WorkflowControlContracts.lossTargetMapping(control.configuration,address:control.authority.address).roles[0].dtype,.float32)
        let(bceOwner,bceControl,bce)=try changed(.bce,key:"weight_mode",value:.string("none")),bceBefore=bceOwner.document
        XCTAssertEqual(bce.draft.roles.map(\.name),["target"]);XCTAssertNotNil(bce.roleChangeMessage)
        XCTAssertEqual(try WorkflowControlContracts.lossTargetMapping(bceControl.configuration,address:bceControl.authority.address).roles.map(\.role),["target","weight"])
        XCTAssertTrue(bce.discard());XCTAssertEqual(bceOwner.document,bceBefore)
        XCTAssertTrue(bce.canApply);XCTAssertTrue(bce.apply())
        XCTAssertEqual(try WorkflowControlContracts.lossTargetMapping(bceControl.configuration,address:bceControl.authority.address).roles.map(\.role),["target"])
    }

}
