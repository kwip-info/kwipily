import XCTest
@testable import KwipMLCore

final class AttentionDeclarationTests:XCTestCase {
    func testNativePortsIntegerMasksNamedWeightsAndParameterlessUtility() throws {
        let factories:[String:(ComponentDefinition)throws->CapabilityDescriptor] = ["multihead_attention":AttentionCapabilityBridge.multiheadAttentionDeclaration,"transformer":AttentionCapabilityBridge.transformerDeclaration,"transformer_encoder":AttentionCapabilityBridge.transformerEncoderDeclaration,"transformer_decoder":AttentionCapabilityBridge.transformerDecoderDeclaration,"scaled_dot_product_attention":AttentionCapabilityBridge.scaledDotProductAttentionDeclaration,"attention_mask":AttentionCapabilityBridge.attentionMaskDeclaration]
        var lessons=Set<String>()
        for (id,factory) in factories {
            let entry=try XCTUnwrap(BundledAttentionComponents.entries[id]);let c=try factory(entry.definition)
            XCTAssertTrue(CapabilityContractValidator.validate(c).isValid)
            XCTAssertEqual(c.ports.filter {$0.direction == .input}.map(\.id),entry.definition.inputs)
            XCTAssertEqual(c.ports.filter {$0.direction == .output}.map(\.id),entry.definition.outputs)
            XCTAssertEqual(c.fields,entry.definition.fields);XCTAssertEqual(c.devices,[.cpu,.mps])
            for p in c.ports {XCTAssertEqual(p.schema.tensor?.dtype,(entry.inputConditions[p.id] != nil || id == "attention_mask" && p.direction == .output) ? .int64:.float32)}
            XCTAssertFalse(c.lifecycle.state.contains {$0.kind == .inferenceCache || $0.kind == .recurrent})
            XCTAssertEqual(c.lifecycle.state.contains {$0.kind == .parameters},id != "attention_mask" && id != "scaled_dot_product_attention")
            lessons.formUnion(c.learning.lessonIDs)
            var altered=entry.definition;altered.outputs.append("cache");XCTAssertThrowsError(try factory(altered))
        }
        XCTAssertEqual(lessons.count,5)
    }
    func testOptionalExactDiscoveryRefusesForeignMaskAndExecutionPolicies() throws {
        XCTAssertTrue(AttentionRuntimeCapabilities().componentIDs.isEmpty)
        var descriptors:[String:Data]=[:]
        for id in AttentionRuntimeCapabilities.operations {descriptors[id]=try JSONEncoder().encode(BundledAttentionComponents.entries[id]!.runtimeContract)}
        XCTAssertEqual(AttentionRuntimeCapabilities(descriptors:descriptors).componentIDs,AttentionRuntimeCapabilities.operations)
        for field in ["maskConvention","maskBroadcast","parameterOrder","sdpaMPSDropoutExecution","maximumScoreElements"] {
            var object=try XCTUnwrap(JSONSerialization.jsonObject(with:descriptors["scaled_dot_product_attention"]!) as? [String:Any]);object[field]="unreviewed"
            var changed=descriptors;changed["scaled_dot_product_attention"]=try JSONSerialization.data(withJSONObject:object)
            XCTAssertFalse(AttentionRuntimeCapabilities(descriptors:changed).contains("scaled_dot_product_attention"))
            XCTAssertThrowsError(try AttentionRuntimeCapabilities.verify(changed["scaled_dot_product_attention"]!,componentID:"scaled_dot_product_attention"))
        }
        XCTAssertThrowsError(try AttentionRuntimeCapabilities.verify(Data(repeating:32,count:4097),componentID:"multihead_attention"))
    }
}
