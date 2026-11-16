import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class BagInputMappingInspectorTests:XCTestCase {
    private let mappingAddress=WorkflowNodeAddress(domain:.control,localID:"mapping")
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    private func fixture(packed:Bool=false,weighted:Bool=false,objective:Bool=false,scratch:Bool=false)throws->(WorkflowAuthoringStore,WorkflowControlNodeContext) {
        var document=try WorkflowExamples.preparedEmbeddingBagClassifier(device:"cpu")
        for index in document.nodes.indices {
            guard case .modelPrimitive(let kind,let version,var values)=document.nodes[index].payload else {continue}
            if kind == "bag_input" || kind == "embedding_bag" {
                values["layout"] = .string(packed ? "packed":"fixed")
                values["per_sample_weights"] = .bool(weighted)
                if kind == "embedding_bag",weighted {values["mode"] = .string("sum")}
                document.nodes[index].payload = .modelPrimitive(type:kind,configurationVersion:version,config:values)
            }
        }
        for port in (packed ? ["offsets"]:[])+(weighted ? ["per_sample_weights"]:[]) {
            document.edges.append(.init(id:"bag-"+port,from:.init(address:.init(domain:.model,localID:"input"),port:port),to:.init(address:.init(domain:.model,localID:"lookup"),port:port),kind:.tensor))
        }
        var bag:[String:JSONValue] = ["schemaVersion":1,"indicesColumns":.array(["id_a","id_b","id_c"].map(JSONValue.string))]
        if weighted {bag["weightColumns"] = .array(["weight_a","weight_b","weight_c"].map(JSONValue.string))}
        if packed {bag["lengthColumn"] = .string("used_items")}
        var configuration:[String:JSONValue] = ["bagInput":.object(bag),"targetColumn":.string("label")]
        if objective {
            let recipeAddress=WorkflowNodeAddress(domain:.control,localID:"objective")
            var recipeConfiguration=LossLearningTask.crossEntropy.configuration
            // The published bag classifier has two actual output classes.
            recipeConfiguration["class_weights"] = .array([1,1])
            document.nodes.append(.init(address:recipeAddress,name:"Desired class",x:0,y:0,payload:.lossRecipe(type:"loss_cross_entropy",configurationVersion:1,config:recipeConfiguration)))
            document.edges.append(.init(id:"bag-objective",from:.init(address:recipeAddress,port:"out"),to:.init(address:WorkflowExamples.trainAddress,port:"objective"),kind:.objectiveRecipe))
            let roles=LossTargetMapping(componentID:"loss_cross_entropy",inputColumns:["id_a","id_b","id_c"],roles:[.init(role:"target",columns:["label"],sampleShape:[],dtype:.int64)])
            configuration=["bagInput":.object(bag),"lossTargetMapping":try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(roles))]
        }
        let mappingIndex=try XCTUnwrap(document.nodes.firstIndex {$0.address == mappingAddress})
        document.nodes[mappingIndex].payload = .trainingData(config:scratch ? [:]:configuration)
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
        let library=ProjectLibrary(directory:directory);_=try library.saveSource(.workflow(document))
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repository.appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertTrue(owner.select(.init(nodeAddresses:[mappingAddress]),lease:owner.lease()))
        return(owner,try WorkflowControlNodeContext(owner:owner,address:mappingAddress))
    }
    func testStoredAndScratchConnectedBagsSelectBagRouteAndDenseApplyCannotErasePacket()throws {
        for scratch in [false,true] {
            let(_,control)=try fixture(scratch:scratch),before=control.configuration
            XCTAssertTrue(control.requiresBagMapping)
            XCTAssertFalse(control.applyMapping(features:["first","second","third"],target:"label",expected:before))
            XCTAssertEqual(control.configuration,before)
            let editor=try BagTrainingMappingInspectorContext(context:control)
            XCTAssertEqual(editor.draft.rows.count,3)
            if scratch {XCTAssertTrue(editor.draft.rows.allSatisfy {$0.indexColumn.isEmpty});XCTAssertNotNil(editor.validationMessage)}
            else {XCTAssertEqual(editor.draft.rows.map(\.indexColumn),["id_a","id_b","id_c"]);XCTAssertNil(editor.validationMessage)}
        }
    }
    func testFixedEditPendingDiscardApplyAndUndoPreserveExactNames()throws {
        let(owner,control)=try fixture(),before=owner.document,editor=try BagTrainingMappingInspectorContext(context:control)
        XCTAssertTrue(editor.update {$0.rows[0].indexColumn=" renamed, ID "})
        XCTAssertTrue(editor.hasChanges);XCTAssertTrue(owner.hasPendingInspectorEdits)
        XCTAssertTrue(editor.discard());XCTAssertFalse(owner.hasPendingInspectorEdits)
        XCTAssertTrue(editor.update {$0.rows[0].indexColumn=" renamed, ID "});XCTAssertTrue(editor.apply())
        XCTAssertEqual(try WorkflowControlContracts.bagInputMapping(control.configuration,address:mappingAddress).indicesColumns,[" renamed, ID ","id_b","id_c"])
        XCTAssertFalse(owner.hasPendingInspectorEdits)
        XCTAssertTrue(owner.undo(lease:owner.lease()));XCTAssertEqual(owner.document.nodes,before.nodes);XCTAssertEqual(owner.document.edges,before.edges)
        XCTAssertFalse(editor.apply());XCTAssertFalse(editor.discard())
        XCTAssertTrue(owner.redo(lease:owner.lease()))
        let reopened=try WorkflowControlNodeContext(owner:owner,address:mappingAddress)
        XCTAssertNotNil(reopened.configuration["bagInput"])
    }
    func testPackedWeightsStayPairedDuringReorderAndRejectColumnRoleCollisions()throws {
        let(_,control)=try fixture(packed:true,weighted:true),editor=try BagTrainingMappingInspectorContext(context:control)
        XCTAssertEqual(editor.draft.lengthColumn,"used_items")
        XCTAssertTrue(editor.update {$0.move($0.rows[2].id,by:-1)})
        XCTAssertTrue(editor.apply())
        let mapping=try WorkflowControlContracts.bagInputMapping(control.configuration,address:mappingAddress)
        XCTAssertEqual(mapping.indicesColumns,["id_a","id_c","id_b"])
        XCTAssertEqual(mapping.weightColumns,["weight_a","weight_c","weight_b"])
        XCTAssertEqual(mapping.lengthColumn,"used_items")
        let before=control.configuration
        XCTAssertTrue(editor.update {$0.rows[0].weightColumn="label"})
        XCTAssertNotNil(editor.validationMessage);XCTAssertFalse(editor.apply());XCTAssertEqual(control.configuration,before)
        XCTAssertTrue(editor.discard())
        XCTAssertTrue(editor.update {$0.lengthColumn=""});XCTAssertNotNil(editor.validationMessage);XCTAssertFalse(editor.apply())
        XCTAssertTrue(editor.discard())
        XCTAssertTrue(editor.update {$0.rows[1].indexColumn=$0.rows[0].indexColumn});XCTAssertNotNil(editor.validationMessage)
    }
    func testActualObjectiveBagRolesRemainEditableAndInputsDeriveFromPacket()throws {
        let(owner,control)=try fixture(packed:true,weighted:true,objective:true),before=owner.document
        let editor=try LossTargetMappingInspectorContext(context:control)
        XCTAssertNotNil(editor.bagDraft);XCTAssertNil(editor.validationMessage)
        XCTAssertEqual(editor.draft.roles[0].dtype,.int64)
        XCTAssertTrue(editor.updateBag {$0.move($0.rows[2].id,by:-1)})
        XCTAssertTrue(editor.update {$0.roles[0].columns.setName("desired class",for:$0.roles[0].columns.rows[0].id)})
        XCTAssertTrue(editor.apply())
        let bag=try WorkflowControlContracts.bagInputMapping(control.configuration,address:mappingAddress)
        let objective=try WorkflowControlContracts.lossTargetMapping(control.configuration.filter {$0.key != "bagInput"},address:mappingAddress)
        XCTAssertEqual(objective.inputColumns,bag.indicesColumns)
        XCTAssertEqual(objective.roles[0].columns,["desired class"])
        XCTAssertEqual(bag.weightColumns,["weight_a","weight_c","weight_b"])
        XCTAssertTrue(owner.undo(lease:owner.lease()));XCTAssertEqual(owner.document.nodes,before.nodes);XCTAssertEqual(owner.document.edges,before.edges)
        XCTAssertFalse(editor.apply())
    }
    func testBusyAndRetiredBagDraftCannotOverwriteSource()throws {
        let(owner,control)=try fixture(),editor=try BagTrainingMappingInspectorContext(context:control),before=control.configuration
        XCTAssertTrue(editor.update {$0.target="new label"})
        owner.setExecutionBusy(true)
        XCTAssertFalse(editor.apply());XCTAssertFalse(editor.discard());XCTAssertFalse(editor.update {$0.target="blocked"})
        owner.setExecutionBusy(false)
        XCTAssertTrue(owner.select(.init(),lease:owner.lease()))
        XCTAssertFalse(editor.apply());XCTAssertFalse(editor.update {$0.target="retired"})
        guard case .trainingData(let saved)=owner.document.nodes.first(where:{$0.address == mappingAddress})?.payload else {return XCTFail("Keep mapping source")}
        XCTAssertEqual(saved,before)
    }
    func testSourceGeometryChangesRetireDraftAndInvalidCapacityNeverCommits()throws {
        let(owner,control)=try fixture(packed:true,weighted:true),editor=try BagTrainingMappingInspectorContext(context:control),before=control.configuration
        XCTAssertTrue(editor.update {$0.rows.removeLast()});XCTAssertNotNil(editor.validationMessage);XCTAssertFalse(editor.apply())
        XCTAssertTrue(editor.discard());XCTAssertTrue(editor.update {$0.target="pending target"})
        let input=WorkflowNodeAddress(domain:.model,localID:"input")
        XCTAssertTrue(owner.select(.init(nodeAddresses:[input]),lease:owner.lease()))
        let authority=try WorkflowNodeEditingAuthority(owner:owner,address:input)
        var config=authority.configuration;config["maximum_items_per_bag"]=2
        XCTAssertTrue(authority.configure(config,expected:authority.configuration))
        XCTAssertFalse(editor.apply())
        guard case .trainingData(let saved)=owner.document.nodes.first(where:{$0.address == mappingAddress})?.payload else {return XCTFail("Keep mapping source")}
        XCTAssertEqual(saved,before)
    }
}
