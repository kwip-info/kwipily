import XCTest
@testable import KwipMLCore

final class WorkflowMixtureContainerTests:XCTestCase {
    private let root=WorkflowNodeAddress(domain:.model,localID:"mixture")
    func testExistingMixtureProjectsExposeEveryChainWithoutChangingExecutionSource()throws {
        let source=try MixtureExpertsPractice.document(),registry=try ComponentRegistry.builtIn()
        let definition=try WorkflowInlineContainer.definition(in:source,root:root)
        XCTAssertEqual(definition.body.nodes.count,12)
        XCTAssertEqual(definition.body.nodes.filter{$0.type=="input"}.count,0)
        XCTAssertEqual(definition.body.nodes.filter{$0.type=="output"}.count,0)
        let data=try DataFlowOperatorRegistry.load(from:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runner/dataflow-operators.json"))
        let base=try WorkflowCanvasProjector.project(source,modelRegistry:registry,dataRegistry:data)
        let projected=try base.addingInlineMembers(source,expandedRoots:[root],modelRegistry:registry,dataRegistry:data)
        XCTAssertEqual(projected.graph.nodes.count,base.graph.nodes.count+12)
        XCTAssertFalse(definition.body.nodes.contains{$0.id.hasPrefix("$router/")})
        XCTAssertEqual(definition.body.edges.count,8)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(source)),source)
    }
    func testInlineChangesOneExpertAndPreservesStableParameterIdentities()throws {
        let source=try MixtureExpertsPractice.document(),registry=try ComponentRegistry.builtIn()
        let before=try XCTUnwrap(source.nodes.first{$0.address==root})
        let changed=try WorkflowInlineContainer.apply(.configure(childID:"expert_a/hidden",config:["in_features":2,"out_features":8,"bias":false]),to:source,root:root,registry:registry)
        guard case .modelPrimitive(_,_,let old)=before.payload,case .modelPrimitive(_,_,let values)=changed.nodes.first(where:{$0.address==root})!.payload else{return XCTFail("MoE remains routed primitive")}
        XCTAssertEqual(old["router"],values["router"]);XCTAssertEqual(old["expert_2"],values["expert_2"])
        let parsed=try MixtureExpertsConfiguration(values:values,registry:registry)
        XCTAssertEqual(parsed.experts[0].id,"expert_a")
        XCTAssertEqual(parsed.experts[0].graph.nodes.first{$0.id=="hidden"}?.config["bias"],false)
        XCTAssertTrue(changed.definitions==source.definitions,"No eager composite or duplicate source is persisted")
    }
    func testDisconnectIsEditableButExecutionRemainsStrictAndCrossExpertWiresAreRefused()throws {
        let source=try MixtureExpertsPractice.document(),registry=try ComponentRegistry.builtIn()
        let definition=try WorkflowInlineContainer.definition(in:source,root:root)
        let wire=try XCTUnwrap(definition.body.edges.first{$0.source=="expert_a/hidden"})
        let changed=try WorkflowInlineContainer.apply(.disconnect(edgeID:wire.id),to:source,root:root,registry:registry)
        _=try WorkflowCodec.encode(changed)
        guard case .modelPrimitive(_,_,let values)=changed.nodes.first(where:{$0.address==root})!.payload else{return XCTFail()}
        XCTAssertThrowsError(try MixtureExpertsConfiguration(values:values,registry:registry))
        XCTAssertEqual(try WorkflowInlineContainer.definition(in:changed,root:root).body.nodes.count,12)
        XCTAssertThrowsError(try WorkflowInlineContainer.apply(.connect(from:.init("expert_a/hidden","out"),to:.init("expert_b/activation","in")),to:source,root:root,registry:registry))
        let fixed=try WorkflowInlineContainer.apply(.connect(from:.init(wire.source,wire.sourcePort),to:.init(wire.target,wire.targetPort)),to:changed,root:root,registry:registry)
        guard case .modelPrimitive(_,_,let restored)=fixed.nodes.first(where:{$0.address==root})!.payload else{return XCTFail()}
        _=try MixtureExpertsConfiguration(values:restored,registry:registry)
        XCTAssertThrowsError(try WorkflowInlineContainer.apply(.delete(childID:"expert_a/input"),to:source,root:root,registry:registry))
    }
    func testQualifiedBoundaryLabelsDoNotRewriteAuthoredNamesAndSourceIndexDoesNotStale()throws {
        let source=try MixtureExpertsPractice.document(),registry=try ComponentRegistry.builtIn()
        let address=WorkflowInlineContainer.address(root:root,childID:"expert_a/hidden")
        XCTAssertEqual(try WorkflowInlineContainer.parse(in:source,address:address)?.childID,"expert_a/hidden")
        let renamed=try WorkflowInlineContainer.apply(.rename(childID:"expert_a/hidden",name:"Observations"),to:source,root:root,registry:registry)
        XCTAssertEqual(try WorkflowInlineContainer.node(in:renamed,root:root,childID:"expert_a/hidden",registry:registry).name,"Observations")
        XCTAssertEqual(try WorkflowInlineContainer.definition(in:renamed,root:root).body.nodes.first{$0.id=="expert_a/hidden"}?.name,"Observations")
        XCTAssertTrue(renamed.matchesCapturedConstruction(source))
        var removed=source;removed.nodes.removeAll{$0.address==root}
        XCTAssertNil(try WorkflowInlineContainer.parse(in:removed,address:address))
        XCTAssertNotNil(try WorkflowInlineContainer.parse(in:source,address:address))
    }

}
