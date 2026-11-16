import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class InitializationRecipeInspectorTests:XCTestCase {
    private func fixture()throws->(WorkflowAuthoringStore,WorkflowNodeEditingAuthority,TrainInlineRecipeDraft) {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")),registry=try ComponentRegistry.builtIn()
        var document=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        let model=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:registry,dataRegistry:data,contractVersion:7).model
        let node=try XCTUnwrap(model.graph.nodes.first {$0.type == "linear"})
        let config=try InitializationConfiguration(values:["seed":42,"rules":.array([.object(["owner":.object(["node":.string(node.id),"parameter":.string("weight")]),"algorithm":.string("normal"),"settings":.object(["mean":0,"std":0.2])])])])
        let address=WorkflowExamples.trainAddress
        document.schemaVersion=13
        let index=try XCTUnwrap(document.nodes.firstIndex {$0.address==address})
        guard case .train(var train)=document.nodes[index].payload else {throw GraphConfigurationError("Train fixture")}
        train["initialization"]=config.recipeValue;document.nodes[index].payload = .train(config:train)
        let library=ProjectLibrary(directory:root);_=try library.saveSource(.workflow(document))
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:registry,dataRegistry:data)
        XCTAssertTrue(owner.select(.init(nodeAddresses:[address]),lease:owner.lease()))
        let authority=try WorkflowNodeEditingAuthority(owner:owner,address:address)
        return(owner,authority,.init(session:.init(authority:authority),role:"initialization"))
    }
    func testRealOwnersExactSeedDraftApplyDiscardAndUndo()throws {
        let(owner,authority,draft)=try fixture(),before=owner.document
        let context=try XCTUnwrap(draft.initialization)
        XCTAssertGreaterThan(context.owners.count,1)
        XCTAssertTrue(context.owners.contains {$0.parameter == "weight" && $0.shape.count==2})
        XCTAssertTrue(context.setSeedText("9223372036854775807"));XCTAssertTrue(context.canApply)
        XCTAssertEqual(owner.document,before);XCTAssertTrue(owner.hasPendingInspectorEdits)
        XCTAssertTrue(context.discard());XCTAssertEqual(context.seedText,"42");XCTAssertFalse(owner.hasPendingInspectorEdits)
        XCTAssertTrue(context.setSeedText("9007199254740993"));XCTAssertFalse(context.apply());XCTAssertTrue(draft.apply())
        XCTAssertEqual(draft.committedConfiguration["seed"],.integer(9007199254740993));XCTAssertTrue(authority.isCurrent)
        XCTAssertFalse(owner.hasPendingInspectorEdits)
        XCTAssertTrue(owner.undo(lease:owner.lease()));XCTAssertEqual(owner.document.nodes.first {$0.address==authority.address}?.payload,before.nodes.first {$0.address==authority.address}?.payload)
    }
    func testOwnerMembershipAlgorithmFieldsAndDormantDraftControls()throws {
        let(_,_,draft)=try fixture()
        let context=try XCTUnwrap(draft.initialization)
        XCTAssertFalse(context.setOwner("foreign",at:0))
        XCTAssertTrue(context.setSetting("std",value:.number(0.35),at:0))
        XCTAssertTrue(context.setAlgorithm("xavier_uniform",at:0))
        XCTAssertFalse(context.setSetting("std",value:0.9,at:0))
        XCTAssertTrue(context.setSetting("gain",value:0.7,at:0))
        XCTAssertTrue(context.setAlgorithm("normal",at:0))
        guard case .object(let settings)=context.rules[0]["settings"] else {return XCTFail("Typed rule settings")}
        XCTAssertEqual(settings["std"],.number(0.35))
        XCTAssertFalse(context.setSetting("std",value:.number(.infinity),at:0))
        XCTAssertTrue(context.addOwner());XCTAssertEqual(context.rules.count,2)
        XCTAssertTrue(context.removeRule(at:1));XCTAssertEqual(context.rules.count,1)
    }
    func testInvalidSeedBusyAndRetiredEditorsCannotCommit()throws {
        let(owner,authority,draft)=try fixture()
        let context=try XCTUnwrap(draft.initialization)
        XCTAssertTrue(context.setSeedText("9223372036854775808"));XCTAssertFalse(context.canApply);XCTAssertNotNil(context.validationMessage)
        XCTAssertFalse(context.apply());XCTAssertTrue(context.discard())
        owner.setExecutionBusy(true);XCTAssertFalse(context.setSeedText("5"));XCTAssertFalse(context.apply());owner.setExecutionBusy(false)
        XCTAssertTrue(context.setSeedText("5"));XCTAssertTrue(owner.hasPendingInspectorEdits)
        authority.retire();XCTAssertFalse(owner.hasPendingInspectorEdits)
        XCTAssertFalse(context.setSeedText("6"));XCTAssertFalse(context.apply());XCTAssertFalse(context.discard())
    }
    func testColdPendingRulesInvalidSeedAndNumericTextKeepOriginalSourceBaseline()throws {
        let(_,_,draft)=try fixture()
        let warm=try XCTUnwrap(draft.initialization)
        let source=warm.draftConfiguration
        var pending=false
        let cold=InitializationRecipeInspectorContext(configuration:source,owners:warm.owners,origin:"cold-context",isCurrent:{true},isBusy:{false},apply:{_,_ in XCTFail("Restoring a draft must not commit");return false},pending:{pending=$0})
        var edited=source
        guard case .array(var rules)=edited["rules"],case .object(var rule)=rules[0],case .object(var settings)=rule["settings"] else {return XCTFail("Fixture rules")}
        settings["std"]=0.35;rule["settings"] = .object(settings);rules[0] = .object(rule);edited["rules"] = .array(rules)
        XCTAssertTrue(cold.restorePending(configuration:edited,seedText:"invalid seed",rawNumericDrafts:["rule.0.std":"-"]))
        XCTAssertEqual(cold.expectedConfiguration,source);XCTAssertEqual(cold.draftConfiguration,edited)
        XCTAssertEqual(cold.seedText,"invalid seed");XCTAssertEqual(cold.rawNumericDrafts["rule.0.std"],"-")
        XCTAssertTrue(pending);XCTAssertFalse(cold.canApply)
        XCTAssertTrue(cold.setSeedText("42"));XCTAssertFalse(cold.canApply,"Uncommitted numeric text still blocks Apply")
        XCTAssertFalse(cold.setRawNumericDraft("rule.0.std",text:"0.4",origin:"foreign"))
        XCTAssertFalse(cold.restorePending(configuration:edited,seedText:"42",rawNumericDrafts:["unknown.field":"7"]))
        XCTAssertTrue(cold.discard());XCTAssertEqual(cold.draftConfiguration,source);XCTAssertTrue(cold.rawNumericDrafts.isEmpty);XCTAssertFalse(pending)
    }

}
