import XCTest
@testable import KwipMLCore

final class BagSourceGraphTests:XCTestCase {
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
        registry.components.removeAll {["bag_input","embedding_bag"].contains($0.id)}
        registry.components += [
            .init(id:"bag_input",title:"Bag input",inputs:[],outputs:["indices","offsets","per_sample_weights"],fields:sourceFields),
            .init(id:"embedding_bag",title:"Embedding bag",inputs:["indices","offsets","per_sample_weights"],outputs:["out"],fields:tableFields)]
        return registry
    }
    private func graph(packed:Bool=false,weighted:Bool=false,terminal:Bool=false)->GraphDocument {
        var source=BagSourceConfiguration.defaults
        source["layout"] = .string(packed ? "packed":"fixed")
        source["per_sample_weights"] = .bool(weighted);source["include_last_offset"] = .bool(terminal)
        var table=EmbeddingBagConfiguration.defaults
        table["layout"] = source["layout"];table["per_sample_weights"] = source["per_sample_weights"]
        table["include_last_offset"] = source["include_last_offset"]
        table["embedding_dim"] = 2;table["mode"] = .string(weighted ? "sum":"mean")
        let ports=["indices"]+(packed ? ["offsets"]:[])+(weighted ? ["per_sample_weights"]:[])
        return .init(nodes:[.init(id:"bags",type:"bag_input",config:source),.init(id:"table",type:"embedding_bag",config:table),.init(id:"output",type:"output")],
                     edges:ports.map {.init(id:$0,source:"bags",sourcePort:$0,target:"table",targetPort:$0)}+[.init(id:"result",source:"table",target:"output")])
    }
    func testExactSourceFieldsBoundariesAndConditionalPorts()throws {
        let fixed=try BagSourceConfiguration(values:BagSourceConfiguration.defaults)
        XCTAssertEqual(fixed.outputPorts,["indices"])
        XCTAssertEqual(fixed.values,BagSourceConfiguration.defaults)
        for bound in [1,128] {
            var values=BagSourceConfiguration.defaults;values["maximum_items_per_bag"] = .integer(Int64(bound))
            XCTAssertEqual(try BagSourceConfiguration(values:values).maximumItemsPerBag,bound)
        }
        var packed=BagSourceConfiguration.defaults;packed["layout"] = .string("packed");packed["per_sample_weights"] = true;packed["include_last_offset"] = true
        XCTAssertEqual(try BagSourceConfiguration(values:packed).outputPorts,["indices","offsets","per_sample_weights"])
        for (key,bad) in [("layout",JSONValue.string("ragged")),("maximum_items_per_bag",.integer(0)),("maximum_items_per_bag",.integer(129)),("maximum_items_per_bag",.number(1.5)),("maximum_items_per_bag",.bool(true)),("per_sample_weights",.integer(1)),("include_last_offset",.null)] {
            var values=BagSourceConfiguration.defaults;values[key]=bad
            XCTAssertThrowsError(try BagSourceConfiguration(values:values),key)
        }
        var missing=BagSourceConfiguration.defaults;missing.removeValue(forKey:"layout")
        XCTAssertThrowsError(try BagSourceConfiguration(values:missing))
        var foreign=BagSourceConfiguration.defaults;foreign["dtype"] = .string("int64")
        XCTAssertThrowsError(try BagSourceConfiguration(values:foreign))
    }
    func testBothLayoutsUseSeparateCoherentCoordinateContractsAndOrdinaryResult()throws {
        let registry=try registry()
        for packed in [false,true] {
            for weighted in [false,true] {
                let graph=graph(packed:packed,weighted:weighted,terminal:packed)
                let result=GraphValidator.validate(graph,registry:registry,context:.explicitObjective)
                XCTAssertTrue(result.isValid,result.issues.map(\.message).joined(separator:"; "))
                let expected=["indices"]+(packed ? ["offsets"]:[])+(weighted ? ["per_sample_weights"]:[])
                XCTAssertEqual(Set(result.outputBagContracts.keys.map(\.portID)),Set(expected))
                XCTAssertEqual(Set(result.outputBagContracts.values.map(\.sourceNodeID)),["bags"])
                XCTAssertTrue(result.outputTensorContracts.keys.allSatisfy {$0.nodeID != "bags"},"Bag coordinates cannot masquerade as dense samples")
                for port in expected {
                    let coordinate=try XCTUnwrap(result.outputBagContracts[.init("bags",port)])
                    XCTAssertEqual(coordinate.role.rawValue,port)
                    XCTAssertEqual(coordinate.dtype,port == "per_sample_weights" ? .float32:.int64)
                    XCTAssertEqual(coordinate.configuration.layout,packed ? "packed":"fixed")
                }
                let output=try XCTUnwrap(result.outputTensorContracts[.init("table","out")])
                XCTAssertEqual(output.layout.sample.dimensions,[2]);XCTAssertEqual(output.dtype,.float32)
                XCTAssertEqual(result.widths["table"],2)
            }
        }
    }
    func testConfiguredPortsAndSchemaTwoMigration()throws {
        let registry=try registry(),source=try XCTUnwrap(registry.component(type:"bag_input")),table=try XCTUnwrap(registry.component(type:"embedding_bag"))
        let fixed=graph(),packed=graph(packed:true,weighted:true,terminal:true)
        XCTAssertEqual(source.configuredOutputs(fixed.nodes[0].config),["indices"])
        XCTAssertEqual(table.configuredInputs(fixed.nodes[1].config),["indices"])
        XCTAssertEqual(source.configuredOutputs(packed.nodes[0].config),["indices","offsets","per_sample_weights"])
        XCTAssertEqual(table.configuredInputs(packed.nodes[1].config),["indices","offsets","per_sample_weights"])
        for node in packed.nodes.prefix(2) {
            XCTAssertEqual(try GraphMigration.effectiveConfiguration(of:node,schemaVersion:2,registry:registry),node.config)
            XCTAssertThrowsError(try GraphMigration.effectiveConfiguration(of:node,schemaVersion:1,registry:registry))
        }
    }
    func testMismatchedRolesFlagsAndForeignSourcesRefuse()throws {
        let registry=try registry()
        for mode in ["role","layout","weights","terminal","foreign"] {
            var graph=graph(packed:true,weighted:true,terminal:true)
            switch mode {
            case "role":graph.edges[1].sourcePort="indices"
            case "layout":graph.nodes[1].config["layout"] = .string("fixed")
            case "weights":graph.nodes[1].config["per_sample_weights"] = false
            case "terminal":graph.nodes[1].config["include_last_offset"] = false
            default:
                var other=graph.nodes[0];other.id="other";graph.nodes.append(other);graph.edges[1].source="other"
            }
            let result=GraphValidator.validate(graph,registry:registry,context:.explicitObjective)
            XCTAssertFalse(result.isValid,mode)
            XCTAssertTrue(result.issues.contains {$0.nodeID == "table"},mode+": "+result.issues.map(\.message).joined(separator:"; "))
            XCTAssertNil(result.outputTensorContracts[.init("table","out")],mode)
        }
    }
    func testDisabledSavedSourcePortAndOrdinaryCoordinateConsumerRefuseExplicitly()throws {
        let registry=try registry()
        var disabled=graph(packed:true)
        disabled.nodes[0].config["layout"] = .string("fixed")
        let bad=GraphValidator.validate(disabled,registry:registry,context:.explicitObjective)
        XCTAssertTrue(bad.issues.contains {$0.nodeID == "bags" && $0.message.lowercased().contains("unavailable")})
        var ordinary=graph()
        ordinary.nodes[1] = .init(id:"table",type:"relu")
        ordinary.edges[0].targetPort="in"
        let result=GraphValidator.validate(ordinary,registry:registry,context:.explicitObjective)
        XCTAssertTrue(result.issues.contains {$0.nodeID == "table" && $0.message.lowercased().contains("bag")})
    }
    func testTableParametersAndFrozenStateUseExistingAggregateBudgets()throws {
        let registry=try registry()
        var frozen=graph();frozen.nodes[1].config["freeze"] = true
        let frozenResult=GraphValidator.validate(frozen,registry:registry,context:.explicitObjective)
        XCTAssertTrue(frozenResult.issues.contains {$0.message.contains("trainable parameters")})
        XCTAssertNotNil(frozenResult.outputTensorContracts[.init("table","out")])
        var oversized=graph();oversized.nodes[1].config["num_embeddings"] = 1_000_001
        let result=GraphValidator.validate(oversized,registry:registry,context:.explicitObjective)
        XCTAssertTrue(result.issues.contains {$0.message.contains("2 million parameters")})
    }
    func testLegacyDenseAndIntegerLookupRemainUnchanged()throws {
        let registry=try registry()
        let result=GraphValidator.validate(ExampleFlows.parallelClassifier,registry:registry)
        XCTAssertTrue(result.isValid,result.issues.map(\.message).joined(separator:"; "))
        XCTAssertTrue(result.outputBagContracts.isEmpty)
        XCTAssertEqual(result.widths["output"],2)
    }
}
