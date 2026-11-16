import XCTest
@testable import KwipMLCore

final class AttentionConfigurationTests: XCTestCase {
    func testDefaultNativeOwnershipAndParameterlessUtilitiesAreIndependent() throws {
        let counts = ["multihead_attention":80,"transformer":448,"transformer_encoder":172,"transformer_decoder":260,"scaled_dot_product_attention":0,"attention_mask":0,"rotary_position":0]
        XCTAssertEqual(AttentionConfiguration.componentIDs, Set(counts.keys))
        for (id,count) in counts {
            let c = try AttentionConfiguration(componentID:id,values:[:])
            XCTAssertEqual(c.parameterCount,count,id)
            XCTAssertFalse(c.stochastic)
            XCTAssertTrue(c.supports(device:"cpu"));XCTAssertTrue(c.supports(device:"mps"));XCTAssertFalse(c.supports(device:"cuda"))
            if id=="multihead_attention" {XCTAssertEqual(c.values["kdim"],.null);XCTAssertEqual(c.values["vdim"],.null);XCTAssertEqual(c.outputPorts,["out","weights"])}
            else if id=="rotary_position" {XCTAssertEqual(c.outputPorts,["query","key"])}
            else if id=="attention_mask" {XCTAssertEqual(c.outputPorts,["mask","padding"])}
            else {XCTAssertEqual(c.outputPorts,["out"])}
        }
    }
    func testUnequalNativeProjectionsAugmentationAndNamedWeightsGeometry() throws {
        let c = try AttentionConfiguration(componentID:"multihead_attention",values:["kdim":3,"vdim":5,"add_bias_kv":true,"add_zero_attn":true,"average_weights":false,"attention_mask_mode":.string("supplied"),"padding_mask_mode":.string("supplied")])
        XCTAssertEqual(c.configuredInputs,["query","key","value","attn_mask","key_padding_mask"])
        XCTAssertEqual(c.parameterShapes.map(\.name),["q_proj_weight","k_proj_weight","v_proj_weight","in_proj_bias","bias_k","bias_v","out_proj.weight","out_proj.bias"])
        XCTAssertEqual(c.parameterShapes.map(\.shape),[[4,4],[4,3],[4,5],[12],[1,1,4],[1,1,4],[4,4],[4]])
        XCTAssertEqual(c.parameterCount,88)
        let r = try c.resolved(layouts:[.init(sample:[2,4]),.init(sample:[3,3]),.init(sample:[3,5]),.init(sample:[2,3]),.init(sample:[3])])
        XCTAssertEqual(r.outputs["out"]?.sample.dimensions,[2,4]);XCTAssertEqual(r.outputs["weights"]?.sample.dimensions,[2,2,5])
        XCTAssertEqual(r.outputDTypes,["out":.float32,"weights":.float32]);XCTAssertEqual(r.outputElementsPerObservation,28);XCTAssertEqual(r.scoreElementsPerObservation,20);XCTAssertEqual(r.maskElementsPerObservation,9);XCTAssertEqual(r.workPerObservation,240)
        var values=c.values;values["return_weights"]=false
        let quiet=try AttentionConfiguration(componentID:c.componentID,values:values)
        XCTAssertEqual(quiet.outputPorts,["out"]);XCTAssertNil(try quiet.resolved(layouts:[.init(sample:[2,4]),.init(sample:[3,3]),.init(sample:[3,5]),.init(sample:[2,3]),.init(sample:[3])]).outputs["weights"])
    }
    func testStacksPreserveSourceRequestedDirectionsAndNativeFinalNormOrder() throws {
        let full=try AttentionConfiguration(componentID:"transformer",values:[:])
        let r=try full.resolved(layouts:[.init(sample:[3,4]),.init(sample:[2,4])])
        XCTAssertEqual(r.outputs["out"]?.sample.dimensions,[2,4]);XCTAssertEqual(r.scoreElementsPerObservation,38)
        XCTAssertEqual(full.parameterShapes.first?.name,"encoder.layers.0.self_attn.in_proj_weight")
        XCTAssertEqual(full.parameterShapes.last?.name,"decoder.norm.bias")
        let encoder=try AttentionConfiguration(componentID:"transformer_encoder",values:["num_layers":2,"bias":false,"final_norm":true,"norm_first":true])
        XCTAssertEqual(encoder.parameterCount,276);XCTAssertEqual(encoder.parameterShapes.last?.name,"norm.weight")
        XCTAssertFalse(encoder.parameterShapes.contains {$0.name.hasSuffix("bias")})
        let decoder=try AttentionConfiguration(componentID:"transformer_decoder",values:["target_mask_mode":.string("causal"),"memory_mask_mode":.string("supplied")])
        XCTAssertEqual(decoder.configuredInputs,["target","memory","memory_mask"])
        XCTAssertEqual(try decoder.resolved(layouts:[.init(sample:[2,4]),.init(sample:[3,4]),.init(sample:[2,3])]).outputs["out"]?.sample.dimensions,[2,4])
    }
    func testSDPAOwnsNoWeightsAndMaskUtilityHasIntegerOutputs() throws {
        let sdpa=try AttentionConfiguration(componentID:"scaled_dot_product_attention",values:["dropout":0.25,"mask_mode":.string("supplied")])
        XCTAssertEqual(sdpa.parameterShapes,[]);XCTAssertTrue(sdpa.stochastic);XCTAssertEqual(sdpa.values["scale"],.null)
        let r=try sdpa.resolved(layouts:[.init(sample:[2,3,4]),.init(sample:[2,5,4]),.init(sample:[2,5,6]),.init(sample:[3,5])])
        XCTAssertEqual(r.outputs["out"]?.sample.dimensions,[2,3,6]);XCTAssertEqual(r.scoreElementsPerObservation,30);XCTAssertEqual(r.workPerObservation,300)
        let mask=try AttentionConfiguration(componentID:"attention_mask",values:["mode":.string("blocked"),"blocked_queries":.array([0,1]),"blocked_keys":.array([1,2]),"padding_keys":.array([0])])
        let mr=try mask.resolved(layouts:[.init(sample:[2,4]),.init(sample:[3,5])])
        XCTAssertEqual(mr.outputDTypes,["mask":.int64,"padding":.int64]);XCTAssertEqual(mr.outputs["mask"]?.sample.dimensions,[2,3]);XCTAssertEqual(mr.outputs["padding"]?.sample.dimensions,[3]);XCTAssertEqual(mr.outputElementsPerObservation,9);XCTAssertEqual(mr.maskElementsPerObservation,9);XCTAssertEqual(mr.workPerObservation,9)
        XCTAssertThrowsError(try mask.resolved(layouts:[.init(sample:[2,2,4]),.init(sample:[3,3,4])]))
    }
    func testMinimumPositiveFloat32AndCanonicalWholeNumbersStayTyped() throws {
        let tiny=Double(Float.leastNonzeroMagnitude)
        let c=try AttentionConfiguration(componentID:"scaled_dot_product_attention",values:["scale":.number(tiny)])
        XCTAssertEqual(c.values["scale"]?.doubleValue,tiny)
        let e=try AttentionConfiguration(componentID:"transformer_encoder",values:["d_model":.number(4),"layer_norm_eps":.number(tiny)])
        XCTAssertEqual(e.values["d_model"],.integer(4));XCTAssertEqual(e.values["layer_norm_eps"]?.doubleValue,tiny)
        XCTAssertThrowsError(try AttentionConfiguration(componentID:"scaled_dot_product_attention",values:["scale":.number(tiny/2)]))
    }
    func testMalformedControlsMaskListsAndStaleAxesRefuse() throws {
        for values in ([["embed_dim":true],["num_heads":3],["kdim":0],["vdim":true],["bias":1],["dropout":1.1],["attention_mask_mode":.string("remember")],["enable_gqa":true],["dropout":.number(.infinity)]] as [[String:JSONValue]]) {XCTAssertThrowsError(try AttentionConfiguration(componentID:"multihead_attention",values:values))}
        for values in ([["scale":0],["scale":true],["scale_mode":.string("automatic")]] as [[String:JSONValue]]) {XCTAssertThrowsError(try AttentionConfiguration(componentID:"scaled_dot_product_attention",values:values))}
        for values in ([["blocked_queries":.array([0]),"blocked_keys":.array([1])],["mode":.string("blocked"),"blocked_queries":.array([0]),"blocked_keys":.array([])],["mode":.string("blocked"),"blocked_queries":.array([0,0]),"blocked_keys":.array([1,1])],["padding_keys":.array([1,1])],["padding_keys":.array([true])],["padding_keys":.array([4096])]] as [[String:JSONValue]]) {XCTAssertThrowsError(try AttentionConfiguration(componentID:"attention_mask",values:values))}
        let mha=try AttentionConfiguration(componentID:"multihead_attention",values:["attention_mask_mode":.string("causal")])
        XCTAssertThrowsError(try mha.resolved(layouts:[.init(sample:[2,4]),.init(sample:[3,4]),.init(sample:[3,4])]))
        let sdpa=try AttentionConfiguration(componentID:"scaled_dot_product_attention",values:[:])
        XCTAssertThrowsError(try sdpa.resolved(layouts:[.init(sample:[2,3,4]),.init(sample:[1,5,4]),.init(sample:[2,5,6])]))
    }
    func testOnlyDeclaredNullableControlsAcceptNull() throws {
        XCTAssertThrowsError(try AttentionConfiguration(componentID:"multihead_attention",values:["dropout":.null]))
        XCTAssertThrowsError(try AttentionConfiguration(componentID:"transformer_encoder",values:["layer_norm_eps":.null]))
        XCTAssertThrowsError(try AttentionConfiguration(componentID:"scaled_dot_product_attention",values:["dropout":.null]))
        XCTAssertNoThrow(try AttentionConfiguration(componentID:"scaled_dot_product_attention",values:["scale":.null]))
        XCTAssertNoThrow(try AttentionConfiguration(componentID:"multihead_attention",values:["kdim":.null,"vdim":.null]))
    }
    func testMaskComparisonGroupsHaveTheSame32GroupCeilingAsRuntime() throws {
        let mask=try AttentionConfiguration(componentID:"attention_mask",values:[:])
        XCTAssertNotNil(try mask.resolved(layouts:[.init(sample:[32,2,4]),.init(sample:[32,3,4])]).outputs["mask"])
        XCTAssertThrowsError(try mask.resolved(layouts:[.init(sample:[33,2,4]),.init(sample:[33,3,4])]))
    }
    func testCheckedScoreWorkMaskAndParameterBudgetsRefuse() throws {
        XCTAssertThrowsError(try AttentionConfiguration(componentID:"transformer_decoder",values:["d_model":512,"nhead":32,"dim_feedforward":4096,"num_layers":8]))
        let c=try AttentionConfiguration(componentID:"multihead_attention",values:[:])
        XCTAssertThrowsError(try c.resolved(layouts:[.init(sample:[4096,4]),.init(sample:[4096,4]),.init(sample:[4096,4])]))
        let mask=try AttentionConfiguration(componentID:"attention_mask",values:[:])
        XCTAssertThrowsError(try mask.resolved(layouts:[.init(sample:[2,4]),.init(sample:[3,3,4])]))
    }
}
