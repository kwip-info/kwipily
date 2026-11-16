import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowModelControlContextTests:XCTestCase {
    private let model=WorkflowNodeAddress(domain:.model,localID:"same")
    private let control=WorkflowNodeAddress(domain:.control,localID:"same")
    private func owner(_ nodes:[WorkflowNode],definitions:[CompositeDefinition]=[])throws->WorkflowAuthoringStore {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
        let library=ProjectLibrary(directory:directory)
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:nodes,definitions:definitions)
        _=try library.saveSource(.workflow(document),id:document.projectID)
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
    }
    func testLegacyFieldEditPreservesRawVersionAndDoesNotInjectBias()throws {
        let original:[String:JSONValue] = ["in_features":.number(2),"out_features":3]
        let store=try owner([.init(address:model,payload:.modelPrimitive(type:"linear",configurationVersion:1,config:original))])
        XCTAssertTrue(store.select(.init(nodeAddresses:[model]),lease:store.lease()))
        let context=try WorkflowModelNodeContext(owner:store,address:model)
        XCTAssertTrue(context.isLegacyBiasEnabled)
        XCTAssertFalse(context.fields.contains {$0.key == "bias"})
        // JSONValue decoding intentionally tries Int64 first. Compare editing against
        // the loaded authoritative values, not the pre-serialization fixture enum.
        let loaded=context.configuration
        XCTAssertEqual(loaded["in_features"],.integer(2))
        XCTAssertTrue(context.setField("out_features",value:4,expected:loaded))
        var expected=loaded;expected["out_features"]=4
        XCTAssertEqual(store.document.nodes[0].payload,.modelPrimitive(type:"linear",configurationVersion:1,config:expected))
        XCTAssertNil(context.configuration["bias"])
        XCTAssertFalse(context.setField("bias",value:false,expected:context.configuration))
        XCTAssertTrue(context.isCurrent)
    }
    func testMissingValuesRemainMissingUntilExplicitCommit()throws {
        let store=try owner([.init(address:model,payload:.modelPrimitive(type:"linear",configurationVersion:2,config:[:]))])
        _=store.select(.init(nodeAddresses:[model]),lease:store.lease())
        let context=try WorkflowModelNodeContext(owner:store,address:model)
        XCTAssertNil(context.value(for:"bias"));XCTAssertNil(context.value(for:"in_features"))
        XCTAssertEqual(context.configuration,[:])
        XCTAssertTrue(context.setField("bias",value:false,expected:[:]))
        XCTAssertEqual(context.configuration,["bias":false])
        XCTAssertNil(context.value(for:"in_features"))
    }
    func testPinnedBlockOverridesCanBeRemovedWithoutChangingPinOrDefinition()throws {
        let definition=CompositeExamples.parallelClassifier.definitions[0]
        let store=try owner([.init(address:model,payload:.modelBlock(reference:definition.reference,overrides:[:]))],definitions:[definition])
        _=store.select(.init(nodeAddresses:[model]),lease:store.lease())
        var opened:WorkflowNodeAddress?
        let context=try WorkflowModelNodeContext(owner:store,address:model,onOpenBlock:{address,lease in
            XCTAssertTrue(store.isCurrent(lease));opened=address
        })
        let inherited=context.value(for:"branchWidth")
        XCTAssertNotNil(inherited)
        XCTAssertTrue(context.setField("branchWidth",value:3,expected:[:]))
        XCTAssertEqual(context.value(for:"branchWidth"),3)
        XCTAssertTrue(context.removeOverride("branchWidth",expected:["branchWidth":3]))
        XCTAssertEqual(context.value(for:"branchWidth"),inherited)
        XCTAssertEqual(store.document.nodes[0].payload,.modelBlock(reference:definition.reference,overrides:[:]))
        XCTAssertEqual(store.document.definitions,[definition])
        XCTAssertTrue(context.openBlock());XCTAssertEqual(opened,model)
        _=store.select(.init(),lease:store.lease())
        XCTAssertFalse(context.openBlock())
    }
    func testPendingModelAndControlDraftsUseSameExecutionGateAndRetireSynchronously()throws {
        let store=try owner([.init(address:model,payload:.modelPrimitive(type:"linear",configurationVersion:2,config:["in_features":2,"out_features":2,"bias":true])),.init(address:control,payload:.runSetup(config:[:]))])
        _=store.select(.init(nodeAddresses:[model]),lease:store.lease())
        let first=try WorkflowModelNodeContext(owner:store,address:model)
        let origin=first.numericOrigin(key:"in_features")
        first.setNumericPending(origin:origin,key:"in_features",pending:true)
        XCTAssertTrue(store.hasPendingInspectorEdits);XCTAssertTrue(store.hasPendingDataEdits,"Legacy capture alias must not bypass non-data drafts.")
        XCTAssertFalse(first.setField("in_features",value:0,expected:first.configuration))
        XCTAssertTrue(store.hasPendingInspectorEdits)
        XCTAssertTrue(first.setField("bias",value:false,expected:first.configuration))
        XCTAssertEqual(first.numericOrigin(key:"in_features"),origin)
        XCTAssertTrue(store.hasPendingInspectorEdits)
        _=store.select(.init(nodeAddresses:[control]),lease:store.lease())
        _=store.select(.init(nodeAddresses:[model]),lease:store.lease())
        XCTAssertFalse(first.isCurrent);XCTAssertFalse(store.hasPendingInspectorEdits)
        first.setNumericPending(origin:origin,key:"in_features",pending:true)
        XCTAssertFalse(store.hasPendingInspectorEdits)
        _=store.select(.init(nodeAddresses:[control]),lease:store.lease())
        let second=try WorkflowControlNodeContext(owner:store,address:control)
        second.setNumericPending(origin:second.numericOrigin(key:"epochs"),key:"epochs",pending:true)
        XCTAssertTrue(store.hasPendingInspectorEdits)
        XCTAssertFalse(second.setRunValue("epochs",value:.number(1.5),expected:[:]))
        XCTAssertTrue(store.hasPendingInspectorEdits)
        XCTAssertTrue(second.setRunValue("epochs",value:2,expected:[:]))
        XCTAssertTrue(store.hasPendingInspectorEdits)
        second.retire();XCTAssertFalse(store.hasPendingInspectorEdits)
    }
    func testMappingPreservesOrderedExactNamesAndRejectsCollisionAtomically()throws {
        let store=try owner([.init(address:control,payload:.trainingData(config:[:]))])
        _=store.select(.init(nodeAddresses:[control]),lease:store.lease())
        let context=try WorkflowControlNodeContext(owner:store,address:control)
        XCTAssertTrue(context.applyMapping(features:[" x, ","second"],target:" target ",expected:[:]))
        let saved=context.configuration
        XCTAssertEqual(saved,["featureColumns":.array([.string(" x, "),.string("second")]),"targetColumn":.string(" target ")])
        XCTAssertFalse(context.applyMapping(features:[" x, "," x, "],target:"target",expected:saved))
        XCTAssertEqual(context.configuration,saved)
        XCTAssertFalse(context.applyMapping(features:["other","second"],target:"target",expected:[:]))
        XCTAssertEqual(context.configuration,saved)
    }
    func testVariableMappingCommitPersistsOrderAndRetiredContextCannotOverwriteIt()throws {
        let store=try owner([.init(address:control,payload:.trainingData(config:[:]))])
        _=store.select(.init(nodeAddresses:[control]),lease:store.lease())
        let context=try WorkflowControlNodeContext(owner:store,address:control)
        XCTAssertTrue(context.applyMapping(features:["third","first","second"],target:"label",expected:[:]))
        let saved=context.configuration
        XCTAssertEqual(saved["featureColumns"],.array(["third","first","second"].map(JSONValue.string)))
        XCTAssertFalse(context.applyMapping(features:["third","first","second"],target:"first",expected:saved))
        XCTAssertEqual(context.configuration,saved)
        _=store.select(.init(),lease:store.lease())
        _=store.select(.init(nodeAddresses:[control]),lease:store.lease())
        XCTAssertFalse(context.applyMapping(features:["replacement"],target:"label",expected:saved))
        let reopened=try WorkflowControlNodeContext(owner:store,address:control)
        XCTAssertEqual(reopened.configuration,saved)
        XCTAssertTrue(reopened.applyMapping(features:["only"],target:"label",expected:saved))
    }
    func testExactPendingBlockDraftCanBeReopenedWithoutPermittingParentMutation()throws {
        let definition=CompositeExamples.parallelClassifier.definitions[0]
        let other=WorkflowNodeAddress(domain:.model,localID:"other")
        let store=try owner([
            .init(address:model,payload:.modelBlock(reference:definition.reference,overrides:[:])),
            .init(address:other,payload:.modelBlock(reference:definition.reference,overrides:[:]))
        ],definitions:[definition])
        _=store.select(.init(nodeAddresses:[model]),lease:store.lease())
        var opened:[WorkflowNodeAddress]=[]
        let callback:(WorkflowNodeAddress,WorkflowEditorLease)->Void = {address,lease in
            XCTAssertTrue(store.isCurrent(lease));opened.append(address)
        }
        let before=try WorkflowModelNodeContext(owner:store,address:model,onOpenBlock:callback)
        XCTAssertTrue(store.beginDraft(root:model,lease:store.lease()))
        XCTAssertFalse(before.isCurrent);XCTAssertFalse(before.openBlock())
        let current=try WorkflowModelNodeContext(owner:store,address:model,onOpenBlock:callback)
        XCTAssertTrue(current.isBusy,"Parent fields must remain locked while editing a child draft.")
        XCTAssertFalse(current.setField("branchWidth",value:3,expected:[:]))
        XCTAssertTrue(current.canOpenBlock)
        XCTAssertTrue(current.openBlock());XCTAssertEqual(opened,[model])
        store.setExecutionBusy(true)
        XCTAssertFalse(current.openBlock());XCTAssertEqual(opened,[model])
        store.setExecutionBusy(false)
        XCTAssertTrue(current.openBlock());XCTAssertEqual(opened,[model,model])
        _=store.select(.init(nodeAddresses:[other]),lease:store.lease())
        let different=try WorkflowModelNodeContext(owner:store,address:other,onOpenBlock:callback)
        XCTAssertFalse(different.canOpenBlock);XCTAssertFalse(different.openBlock())
        _=store.select(.init(nodeAddresses:[model]),lease:store.lease())
        XCTAssertFalse(current.openBlock(),"Away/back must not revive the retired originating callback.")
        XCTAssertEqual(opened,[model,model])
    }

}
