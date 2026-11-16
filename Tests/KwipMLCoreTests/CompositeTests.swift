import XCTest
@testable import KwipMLCore

final class CompositeTests:XCTestCase {
    func expand(_ project:CompositeProjectDocument=CompositeExamples.parallelClassifier)throws->CompositeExpansion {try CompositeExpander.expand(project,registry:.builtIn())}
    func testParallelCompositeFlattensIntoSameArchitectureAndStableOrigins() throws {
        let result=try expand(),original=ExampleFlows.parallelClassifier
        XCTAssertTrue(GraphValidator.validate(result.graph,registry:try .builtIn()).isValid)
        XCTAssertEqual(result.graph.nodes.count,original.nodes.count);XCTAssertEqual(result.graph.edges.count,original.edges.count)
        XCTAssertEqual(result.consumedDefinitions,[CompositeExamples.parallelClassifier.definitions[0].reference])
        XCTAssertEqual(try expand(),result)
        for node in result.graph.nodes {
            let origin=try XCTUnwrap(result.nodeOrigins[node.id])
            let expected=try XCTUnwrap(original.nodes.first {$0.id == origin.localNodeID})
            XCTAssertEqual(node.type,expected.type)
            XCTAssertEqual(node.config, try GraphMigration.effectiveConfiguration(
                of: expected, schemaVersion: original.schemaVersion, registry: .builtIn()))
            XCTAssertFalse(node.id.contains("."));XCTAssertLessThanOrEqual(node.id.count,128)
        }
        let canonical=Set(result.graph.edges.map {"\(result.nodeOrigins[$0.source]!.localNodeID):\($0.sourcePort)>\(result.nodeOrigins[$0.target]!.localNodeID):\($0.targetPort)"})
        XCTAssertEqual(canonical,Set(original.edges.map {"\($0.source):\($0.sourcePort)>\($0.target):\($0.targetPort)"}))
    }
    func testEnvelopeRoundTripAndExplicitInstanceOverrides() throws {
        var project=CompositeExamples.parallelClassifier
        project.root.instances[0].name="My user label"
        project.root.instances[0].overrides=["branchWidth":4]
        let classifier=try XCTUnwrap(project.root.nodes.firstIndex {$0.id == "classifier"})
        project.root.nodes[classifier].config["in_features"]=8
        let decoded=try JSONDecoder().decode(CompositeProjectDocument.self,from:JSONEncoder().encode(project))
        XCTAssertEqual(decoded,project)
        let result=try expand(decoded)
        for node in result.graph.nodes where result.nodeOrigins[node.id]?.localNodeID.hasPrefix("branch-") == true {XCTAssertEqual(node.config["out_features"],4)}
        XCTAssertEqual(project.definitions[0].body.nodes.first {$0.id == "branch-a"}?.config["out_features"],8)
        var wrong=project;wrong.root.instances[0].overrides["typo"]=4
        XCTAssertThrowsError(try expand(wrong))
        wrong=project;wrong.root.instances[0].overrides["branchWidth"]=4.5
        XCTAssertThrowsError(try expand(wrong))
    }
    func testNestedDefinitionAndIndependentInstancesUseDistinctNamespaces() throws {
        var project=CompositeExamples.parallelClassifier
        let inner=project.definitions[0].reference
        let wrapper=CompositeDefinition(reference:CompositeDefinitionReference("wrapper"),title:"Wrapper",body:CompositeGraph(instances:[CompositeInstance(id:"inner",definition:inner)]),inputs:[CompositeInput("in",targets:[CompositePortReference("inner","in")])],outputs:[CompositeOutput("out",source:CompositePortReference("inner","out"))])
        project.definitions.append(wrapper);project.root.instances[0].definition=wrapper.reference
        let result=try expand(project)
        XCTAssertEqual(result.nodeOrigins.values.filter {$0.instancePath == ["features","inner"]}.count,5)
        var two=CompositeExamples.parallelClassifier
        two.root.instances.append(CompositeInstance(id:"second",definition:inner))
        two.root.nodes.append(GraphNode(id:"join",type:"add"))
        two.root.edges.removeAll {$0.id == "outer-out"}
        two.root.edges += [GraphEdge(id:"second-in",source:"input",target:"second"),GraphEdge(id:"first-merge",source:"features",target:"join",targetPort:"a"),GraphEdge(id:"second-merge",source:"second",target:"join",targetPort:"b"),GraphEdge(id:"join-head",source:"join",target:"classifier")]
        let expanded=try expand(two)
        let a=Set(expanded.nodeOrigins.filter {$0.value.instancePath == ["features"]}.map(\.key))
        let b=Set(expanded.nodeOrigins.filter {$0.value.instancePath == ["second"]}.map(\.key))
        XCTAssertEqual(a.count,5);XCTAssertEqual(b.count,5);XCTAssertTrue(a.isDisjoint(with:b))
    }
    func testRecursionMissingRevisionAndAmbiguousBoundaryAreRejected() throws {
        var project=CompositeExamples.parallelClassifier
        project.definitions[0].body.instances.append(CompositeInstance(id:"recursive",definition:project.definitions[0].reference))
        XCTAssertThrowsError(try expand(project))
        project=CompositeExamples.parallelClassifier
        project.root.instances[0].definition.version=ContractVersion(9)
        XCTAssertThrowsError(try expand(project))
        project=CompositeExamples.parallelClassifier
        project.definitions[0].inputs.append(CompositeInput("ambiguous",targets:[CompositePortReference("branch-a","in")]))
        XCTAssertThrowsError(try expand(project))
        project=CompositeExamples.parallelClassifier
        project.definitions[0].outputs[0].source=CompositePortReference("missing","out")
        XCTAssertThrowsError(try expand(project))
    }
    func testPinnedRevisionsRequireExplicitUpgradeAndCannotBeReplaced() throws {
        let original=CompositeExamples.parallelClassifier
        var changed=original.definitions[0];changed.title="Changed under same pin"
        XCTAssertThrowsError(try CompositeProjectEdits.addingRevision(changed,to:original))
        changed.reference.version=ContractVersion(1,1)
        let added=try CompositeProjectEdits.addingRevision(changed,to:original)
        XCTAssertEqual(added.definitions.count,2)
        XCTAssertEqual(added.root.instances[0].definition,original.root.instances[0].definition)
        let upgraded=try CompositeProjectEdits.pinningInstance("features",to:changed.reference,in:added)
        XCTAssertEqual(upgraded.root.instances[0].definition,changed.reference)
        XCTAssertEqual(try expand(upgraded).consumedDefinitions,[changed.reference])
    }
    func testLayoutAndLabelsDoNotChangeModelTopologyOrValues() throws {
        var changed=CompositeExamples.parallelClassifier
        changed.root.instances[0].name="Other meaning";changed.root.instances[0].x=888
        changed.definitions[0].body.nodes[0].name="My projection"
        let a=try expand(),b=try expand(changed)
        XCTAssertEqual(a.graph.edges,b.graph.edges)
        XCTAssertEqual(a.graph.nodes.map(\.id),b.graph.nodes.map(\.id))
        XCTAssertEqual(a.graph.nodes.map(\.config),b.graph.nodes.map(\.config))
        XCTAssertEqual(a.nodeOrigins,b.nodeOrigins)
    }
}

extension CompositeTests {
    func testMalformedDefinitionDefaultRejectedEvenWhenInstanceOverridesIt() throws {
        var project=CompositeExamples.parallelClassifier
        project.definitions[0].parameters[0].field.defaultValue = .number(.nan)
        project.root.instances[0].overrides=["branchWidth":8]
        XCTAssertThrowsError(try expand(project))
    }
    func testMutualRecursionDeepExpansionAndInvalidEnvelopeFail() throws {
        var project=CompositeExamples.parallelClassifier
        let a=CompositeDefinitionReference("cycle-a"),b=CompositeDefinitionReference("cycle-b")
        project.definitions += [CompositeDefinition(reference:a,title:"A",body:CompositeGraph(instances:[CompositeInstance(id:"b",definition:b)]),inputs:[],outputs:[]),CompositeDefinition(reference:b,title:"B",body:CompositeGraph(instances:[CompositeInstance(id:"a",definition:a)]),inputs:[],outputs:[])]
        XCTAssertThrowsError(try expand(project))
        project=CompositeExamples.parallelClassifier
        var previous=project.definitions[0].reference
        for i in 0..<9 {
            let wrapper=CompositeDefinition(reference:CompositeDefinitionReference("depth-\(i)"),title:"Wrapper",body:CompositeGraph(instances:[CompositeInstance(id:"child",definition:previous)]),inputs:[CompositeInput("in",targets:[CompositePortReference("child","in")])],outputs:[CompositeOutput("out",source:CompositePortReference("child","out"))])
            project.definitions.append(wrapper);previous=wrapper.reference
        }
        project.root.instances[0].definition=previous
        XCTAssertThrowsError(try expand(project))
        project=CompositeExamples.parallelClassifier;project.schemaVersion=99
        XCTAssertThrowsError(try expand(project))
    }
    func testExposedNumericParametersDoNotSilentlyRepairDownstreamWidths() throws {
        var project=CompositeExamples.parallelClassifier
        project.root.instances[0].overrides=["branchWidth":4]
        XCTAssertThrowsError(try expand(project))
        XCTAssertEqual(project.root.nodes.first {$0.id == "classifier"}?.config["in_features"],16)
        project=CompositeExamples.parallelClassifier
        project.definitions[0].parameters.append(CompositeParameter(ComponentField(key:"other",title:"Other",defaultValue:8),targets:[CompositeParameterTarget("branch-a","out_features")]))
        XCTAssertThrowsError(try expand(project))
    }
}
