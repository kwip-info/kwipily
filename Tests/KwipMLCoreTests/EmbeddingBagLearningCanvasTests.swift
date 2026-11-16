import XCTest
@testable import KwipMLCore

final class EmbeddingBagLearningCanvasTests:XCTestCase {
    private func registry()throws->ComponentRegistry {
        var registry=try ComponentRegistry.builtIn()
        // Independent descriptor fixtures retain the shared field gate: these are
        // real declared controls, not empty templates that bypass configuration validation.
        func choice(_ key:String,_ value:String,_ options:[String])->ComponentField {
            .init(key:key,title:key,kind:.choice,defaultValue:.string(value),min:nil,max:nil,
                  choices:options.map {.init(value:$0,title:$0)})
        }
        func flag(_ key:String)->ComponentField {
            .init(key:key,title:key,kind:.boolean,defaultValue:false,min:nil,max:nil)
        }
        let sourceFields:[ComponentField]=[
            choice("layout","fixed",["fixed","packed"]),
            .init(key:"maximum_items_per_bag",title:"Maximum items",defaultValue:4,min:1,max:128),
            flag("per_sample_weights"),flag("include_last_offset")]
        let tableFields:[ComponentField]=[
            .init(key:"num_embeddings",title:"Table rows",defaultValue:16,min:1,max:100_000_000),
            .init(key:"embedding_dim",title:"Dimensions",defaultValue:4,min:1,max:4096),
            choice("mode","mean",["mean","sum","max"]),
            .init(key:"padding_idx",title:"Padding",defaultValue:.null,min:-100_000_000,max:99_999_999,allowsNull:true),
            .init(key:"max_norm",title:"Maximum norm",kind:.number,defaultValue:.null,min:1.1754943508222875e-38,max:3.4028234663852886e38,allowsNull:true),
            .init(key:"norm_type",title:"Norm order",kind:.number,defaultValue:2,min:1.1754943508222875e-38,max:3.4028234663852886e38),
            flag("scale_grad_by_freq"),flag("sparse"),flag("include_last_offset"),flag("freeze"),
            choice("layout","fixed",["fixed","packed"]),flag("per_sample_weights")]
        registry.components.removeAll { ["bag_input","embedding_bag"].contains($0.id) }
        registry.components += [
            .init(id:"bag_input",title:"Bag input",inputs:[],outputs:["indices","offsets","per_sample_weights"],fields:sourceFields),
            .init(id:"embedding_bag",title:"Embedding bag",inputs:["indices","offsets","per_sample_weights"],outputs:["out"],fields:tableFields)]
        return registry
    }

    func testRestrictedPaletteUsesCallerDefinitionsAndRefusesUnavailablePublication()throws {
        let base=try registry(),restricted=try EmbeddingBagLearningCanvas.registry(from:base)
        XCTAssertEqual(restricted.components.map(\.id),["bag_input","embedding_bag","output"])
        XCTAssertNil(restricted.component(type:"input"));XCTAssertNil(restricted.component(type:"reshape"))
        XCTAssertEqual(restricted.component(type:"bag_input"),base.component(type:"bag_input"))
        XCTAssertEqual(restricted.component(type:"embedding_bag"),base.component(type:"embedding_bag"))
        XCTAssertTrue(restricted.component(type:"output")!.fields.isEmpty)
        var missing=base;missing.components.removeAll {$0.id == "embedding_bag"}
        XCTAssertThrowsError(try EmbeddingBagLearningCanvas.registry(from:missing))
        var duplicate=base;duplicate.components.append(base.component(type:"bag_input")!)
        XCTAssertThrowsError(try EmbeddingBagLearningCanvas.registry(from:duplicate))
    }
    func testAllSixGraphsResolveTypedBagRolesWithoutDensePackedShape()throws {
        let restricted=try EmbeddingBagLearningCanvas.registry(from:registry())
        for task in EmbeddingBagLearningTask.allCases {
            let graph=try EmbeddingBagExamples.lessonGraph(task:task)
            let result=EmbeddingBagLearningCanvas.inspect(graph,registry:restricted)
            XCTAssertTrue(result.isValid,result.issues.map(\.message).joined(separator:"; "))
            XCTAssertEqual(result.outputTensorContracts[.init("combine","out")]?.dtype,.float32)
            XCTAssertEqual(result.outputTensorContracts[.init("combine","out")]?.layout.sample.dimensions,[3])
            XCTAssertEqual(result.outputTensorContracts[.init("result","out")]?.layout.sample.dimensions,[3])
            XCTAssertNil(result.tensorLayouts["bags"],"Packed and fixed bag source roles are packet coordinates, never ordinary sample layouts")
            for role in try task.sourceConfiguration.outputPorts {
                XCTAssertEqual(result.outputBagContracts[.init("bags",role)]?.role.rawValue,role)
            }
        }
    }
    func testPartialProjectionKeepsSourceRolesAndExplainsIncompleteConstruction()throws {
        let restricted=try EmbeddingBagLearningCanvas.registry(from:registry())
        let blank=EmbeddingBagLearningCanvas.inspect(GraphDocument(),registry:restricted)
        XCTAssertFalse(blank.isValid)
        XCTAssertEqual(blank.issues.count,Set(blank.issues.map(\.message)).count)
        var partial=try EmbeddingBagExamples.lessonGraph(task:.packedWeightedSum)
        partial.nodes=Array(partial.nodes.prefix(2));partial.edges=[]
        let projection=EmbeddingBagLearningCanvas.inspect(partial,registry:restricted)
        XCTAssertFalse(projection.isValid)
        XCTAssertNotNil(projection.outputBagContracts[.init("bags","indices")])
        XCTAssertNotNil(projection.outputBagContracts[.init("bags","offsets")])
        XCTAssertNotNil(projection.outputBagContracts[.init("bags","per_sample_weights")])
        XCTAssertNil(projection.tensorLayouts["bags"])
        XCTAssertNoThrow(try EmbeddingBagLearningSubmission.validatedGraph(partial,complete:false))
    }
    func testTooltipsDistinguishFixedPackedOffsetsWeightsAndReducedOutput()throws {
        let fixed=try EmbeddingBagExamples.lessonGraph(task:.fixedSum)
        let packed=try EmbeddingBagExamples.lessonGraph(task:.packedWeightedSum)
        func descriptor(_ graph:GraphDocument,_ node:String,_ port:String,_ direction:PortDirection = .output)->CanvasPortDescriptor {
            EmbeddingBagLearningCanvas.portDescriptor(graph,nodeID:node,port:port,direction:direction)
        }
        let fixedIDs=descriptor(fixed,"bags","indices")
        XCTAssertEqual(fixedIDs.kind,.tensor)
        XCTAssertTrue(fixedIDs.declared.contains("[bags, items]"))
        XCTAssertTrue(fixedIDs.declared.contains("int64"))
        let packedIDs=descriptor(packed,"bags","indices")
        XCTAssertTrue(packedIDs.declared.contains("[items]"))
        XCTAssertTrue(packedIDs.accessibilityDescription.contains("No observation axis"))
        XCTAssertTrue(packedIDs.accessibilityDescription.contains("unavailable"))
        let offsets=descriptor(packed,"bags","offsets")
        XCTAssertTrue(offsets.declared.contains("[bags + 1]"))
        XCTAssertTrue(offsets.accessibilityDescription.contains("empty"))
        let weights=descriptor(packed,"combine","per_sample_weights",.input)
        XCTAssertTrue(weights.declared.contains("float32"))
        XCTAssertTrue(weights.accessibilityDescription.contains("paired"))
        XCTAssertTrue(weights.accessibilityDescription.contains("gradient"))
        XCTAssertEqual(descriptor(fixed,"bags","offsets").kind,.unknown)
        XCTAssertEqual(descriptor(fixed,"combine","per_sample_weights",.input).kind,.unknown)
        XCTAssertTrue(descriptor(packed,"combine","out").declared.contains("[bags, 3]"))
        XCTAssertTrue(descriptor(packed,"result","in",.input).declared.contains("[bags, 3]"))
        XCTAssertTrue([fixedIDs,packedIDs,offsets,weights].allSatisfy {$0.measured.isEmpty})
    }
    func testSharedCommandsBuildNamedRolesAndRejectCrossedOrDisabledPorts()throws {
        let restricted=try EmbeddingBagLearningCanvas.registry(from:registry())
        for task in EmbeddingBagLearningTask.allCases {
            let full=try EmbeddingBagExamples.lessonGraph(task:task)
            var state=GraphEditorState(document:GraphDocument(nodes:full.nodes))
            for edge in full.edges {
                state=try GraphCommandEngine.apply(.connect(source:.init(edge.source,edge.sourcePort),target:.init(edge.target,edge.targetPort)),to:state,registry:restricted,connectionSemantics:{edge,graph in
                    var candidate=graph;candidate.edges.append(edge)
                    _ = try EmbeddingBagLearningSubmission.validatedGraph(candidate,complete:false)
                }).after
            }
            XCTAssertTrue(EmbeddingBagLearningCanvas.inspect(state.document,registry:restricted).isValid)
            XCTAssertThrowsError(try GraphCommandEngine.apply(.connect(source:.init("bags","out"),target:.init("combine","indices")),to:state,registry:restricted))
        }
        var crossed=try EmbeddingBagExamples.lessonGraph(task:.packedWeightedSum)
        crossed.edges[0].sourcePort="offsets"
        XCTAssertFalse(EmbeddingBagLearningCanvas.inspect(crossed,registry:restricted).isValid)
        var disabled=try EmbeddingBagExamples.lessonGraph(task:.fixedSum)
        disabled.edges[0].sourcePort="per_sample_weights"
        XCTAssertFalse(EmbeddingBagLearningCanvas.inspect(disabled,registry:restricted).isValid)
    }
}
