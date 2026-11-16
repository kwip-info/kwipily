import XCTest
@testable import KwipMLCore

final class BagCompositionSourceTests: XCTestCase {
    func registry()throws->ComponentRegistry {
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
        registry.components.removeAll {["bag_input","embedding_bag"].contains($0.id)}
        registry.components += [
            .init(id:"bag_input",title:"Bag input",inputs:[],outputs:["indices","offsets","per_sample_weights"],fields:sourceFields),
            .init(id:"embedding_bag",title:"Embedding bag",inputs:["indices","offsets","per_sample_weights"],outputs:["out"],fields:tableFields)]
        return registry
    }

    func project(kind: CompositionPolicy.Kind = .graph, packed: Bool = false, weighted: Bool = false) -> CompositeProjectDocument {
        var source = BagSourceConfiguration.defaults
        source["layout"] = .string(packed ? "packed" : "fixed")
        source["maximum_items_per_bag"] = 3
        source["per_sample_weights"] = .bool(weighted)
        source["include_last_offset"] = .bool(packed)
        var table = EmbeddingBagConfiguration.defaults
        table["num_embeddings"] = 5; table["embedding_dim"] = 2
        table["mode"] = .string(weighted ? "sum" : "mean")
        for key in ["layout", "per_sample_weights", "include_last_offset"] { table[key] = source[key] }
        let ports = ["indices"] + (packed ? ["offsets"] : []) + (weighted ? ["per_sample_weights"] : [])
        let ref = CompositeDefinitionReference("bag")
        let definition = CompositeDefinition(reference: ref, title: "Bag lookup", body: .init(nodes: [.init(id: "table", type: "embedding_bag", config: table)]), inputs: ports.map { .init("incoming_" + $0, targets: [.init("table", $0)]) }, outputs: [.init("result", source: .init("table", "out"))], composition: .init(kind: kind, children: [.init(localID: "table", key: kind == .namedMap ? "lookup" : nil)]))
        return .init(schemaVersion: 3, root: .init(nodes: [.init(id: "bags", type: "bag_input", config: source), .init(id: "output", type: "output")], instances: [.init(id: "block", definition: ref)], edges: ports.map { .init(id: $0, source: "bags", sourcePort: $0, target: "block", targetPort: "incoming_" + $0) } + [.init(id: "result", source: "block", sourcePort: "result", target: "output")]), definitions: [definition])
    }
    func testActualSourceExpansionKeepsBagRootOutsideNativeChildrenAndSeparateNestedRoles() throws {
        let registry = try registry()
        for kind in [CompositionPolicy.Kind.graph, .orderedList, .namedMap] {
            for packed in [false, true] {
                for weighted in [false, true] {
                    let source = project(kind: kind, packed: packed, weighted: weighted)
                    let expansion = try CompositeExpander.expand(source, registry: registry)
                    let plan = try XCTUnwrap(expansion.graph.compositionPlan)
                    try plan.validate(graph: expansion.graph, registry: registry)
                    let ports = ["indices"] + (packed ? ["offsets"] : []) + (weighted ? ["per_sample_weights"] : [])
                    XCTAssertEqual(plan.containers.count, 2)
                    XCTAssertEqual(plan.containers[0].inputs.map(\.id), ["bags"])
                    let rootInput = try XCTUnwrap(plan.containers[0].inputs.first)
                    XCTAssertEqual(rootInput.targets.map(\.port), ports)
                    XCTAssertEqual(plan.containers[0].children.map(\.kind), [.container])
                    XCTAssertEqual(plan.containers[1].inputs.map(\.id), ports.map { "incoming_" + $0 })
                    XCTAssertEqual(plan.containers[1].kind, kind)
                    let captured = CompositeSourceSnapshot(project: source, expansion: expansion)
                    XCTAssertEqual(try captured.verifiedExpansion(registry: registry).graph, expansion.graph)
                    let shapes = GraphValidator.validate(expansion.graph, registry: registry, context: .explicitObjective)
                    XCTAssertTrue(shapes.isValid, shapes.issues.map(\.message).joined(separator: "; "))
                    XCTAssertEqual(shapes.outputBagContracts.count, ports.count)
                    XCTAssertNil(shapes.outputTensorContracts[.init("bags", "indices")])
                }
            }
        }
    }
    func testSequenceRemainsUnaryAndInternalBagSourceAlwaysRefuses() throws {
        let registry = try registry()
        _ = try CompositeExpander.expand(project(kind: .sequential), registry: registry)
        for (packed, weighted) in [(true, false), (false, true), (true, true)] {
            XCTAssertThrowsError(try CompositeExpander.expand(project(kind: .sequential, packed: packed, weighted: weighted), registry: registry))
        }
        for policy in [true, false] {
            var source = project()
            source.definitions[0].body.nodes.append(.init(id: "hidden", type: "bag_input", config: BagSourceConfiguration.defaults))
            if !policy { source.definitions[0].composition = nil }
            XCTAssertThrowsError(try CompositeExpander.expand(source, registry: registry))
        }
    }
    func testNestedBoundaryCannotMergeDistinctActualBagCoordinates() throws {
        let registry = try registry()
        let graph = try CompositeExpander.expand(project(packed: true, weighted: true), registry: registry).graph
        let originalPlan = try XCTUnwrap(graph.compositionPlan)
        var containers = originalPlan.containers
        let original = containers[1]
        containers[1] = .init(id: original.id, kind: original.kind, path: original.path, children: original.children, inputs: [.init(id: "merged", targets: original.inputs.flatMap(\.targets))], outputs: original.outputs)
        let plan = CompositionPlan(containers: containers)
        XCTAssertThrowsError(try plan.validate(graph: graph, registry: registry))
    }
}
