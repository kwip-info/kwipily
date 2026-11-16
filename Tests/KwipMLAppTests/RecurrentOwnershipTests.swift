import XCTest
import KwipMLCore
@testable import KwipMLApp

final class RecurrentOwnershipTests:XCTestCase {
    private func fixture(_ id:String,extended:Bool=false)throws->(GraphDocument,OptimizerRecipe,[String:JSONValue]) {
        var graph=try ExampleFlows.recurrentClassifier(componentID:id)
        let cell=id.hasSuffix("_cell"),gates=id.hasPrefix("lstm") ? 4:id.hasPrefix("gru") ? 3:1
        var hidden=2,projected=2,layers=1,directions=1,bias=true
        if extended {
            hidden=4;projected=id=="lstm" ? 2:4;layers=2;directions=2;bias=false
            let i=graph.nodes.firstIndex {$0.type==id}!
            graph.nodes[i].config.merge(["hidden_size":4,"num_layers":2,"bidirectional":true,"bias":false]) {_,new in new}
            if id=="lstm" {graph.nodes[i].config["proj_size"]=2}
            graph.nodes[graph.nodes.firstIndex {$0.type=="linear"}!].config["in_features"] = .integer(Int64(2*directions*projected))
        }
        var ownership:[JSONValue]=[]
        func append(_ name:String,_ shape:[Int]) {ownership.append(.object(["name":.string(name),"shape":.array(shape.map {.integer(Int64($0))}),"dtype":.string("float32")]))}
        for layer in 0..<layers {for direction in 0..<directions {
            let suffix=cell ? "":"_l\(layer)"+(direction==1 ? "_reverse":"")
            let width=layer==0 ? (cell ? 2:1):directions*projected
            append("layers.memory.weight_ih"+suffix,[gates*hidden,width]);append("layers.memory.weight_hh"+suffix,[gates*hidden,projected])
            if bias {append("layers.memory.bias_ih"+suffix,[gates*hidden]);append("layers.memory.bias_hh"+suffix,[gates*hidden])}
            if extended && id=="lstm" {append("layers.memory.weight_hr"+suffix,[projected,hidden])}
        }}
        append("layers.head.weight",[2,cell ? 2:2*directions*projected]);append("layers.head.bias",[2])
        let recipe=OptimizerRecipe(configuration:try .init(componentID:"optimizer_sgd",values:[:]),source:.init(domain:.control,localID:"adjustments"))
        let own:JSONValue = .array(ownership)
        let manifest:[String:JSONValue] = ["settings":.object(["optimizer":.string("SGD"),"optimizerRecipe":recipe.configuration.recipeValue,"optimizerRateSource":.string("explicitRecipe"),"learningRate":.number(recipe.configuration.learningRate)]),"runtime":.object(["torch":.string("2.14.1"),"device":.string("cpu")]),"provenance":.object(["optimizer":.object(["recipe":recipe.configuration.recipeValue,"parameterOwnership":own,"implementation":.string("SGD"),"torchVersion":.string("2.14.1")])]),"checkpoint":.object(["contract":.object(["optimizer":.object(["schemaVersion":1,"recipe":recipe.configuration.recipeValue,"parameterOwnership":own])])])]
        return(graph,recipe,manifest)
    }
    func testIndependentNativeGateLayerReverseProjectionAndBiasOwnership()throws {
        for id in RecurrentConfiguration.componentIDs {
            let(g,r,m)=try fixture(id);XCTAssertNoThrow(try WorkflowLifecycleVerifier.validateOptimizer(m,expected:r,graph:g))
        }
        for id in ["rnn","lstm","gru"] {
            let(g,r,m)=try fixture(id,extended:true);XCTAssertNoThrow(try WorkflowLifecycleVerifier.validateOptimizer(m,expected:r,graph:g))
        }
    }
    func testReorderedLearnedStateAndPretendedHiddenParametersRefuse()throws {
        for id in RecurrentConfiguration.componentIDs {
            let(g,r,m)=try fixture(id)
            for mutation in ["reverse","hidden","dtype","shape"] {
                var changed=m
                guard case .object(var p)=changed["provenance"],case .object(var o)=p["optimizer"],case .array(var own)=o["parameterOwnership"],case .object(var first)=own[0] else {return XCTFail("Fixture ownership missing")}
                if mutation=="reverse" {own.reverse()}
                else {if mutation=="hidden" {first["name"] = .string("layers.memory.hidden")};if mutation=="dtype" {first["dtype"] = .string("float64")};if mutation=="shape" {first["shape"] = .array([1])};own[0] = .object(first)}
                o["parameterOwnership"] = .array(own);p["optimizer"] = .object(o);changed["provenance"] = .object(p)
                XCTAssertThrowsError(try WorkflowLifecycleVerifier.validateOptimizer(changed,expected:r,graph:g),id+mutation)
            }
        }
    }
}
