import XCTest
@testable import KwipMLCore

final class SeparableContainerInterfaceTests:XCTestCase {
    func testExistingSeparableProjectsDisplayAsContainerWithoutSourceRewrite()throws {
        let document=try WorkflowExamples.preparedSeparableConv2dClassifier()
        let before=try WorkflowCodec.encode(document),root=WorkflowNodeAddress(domain:.model,localID:"convolution")
        let definition=try WorkflowInlineContainer.definition(in:document,root:root)
        XCTAssertEqual(definition.body.nodes.map(\.type),["depthwise_conv2d","conv2d"])
        XCTAssertEqual(try WorkflowInlineContainer.node(in:document,root:root,childID:"depthwise",registry:.builtIn()).address,WorkflowInlineContainer.address(root:root,childID:"depthwise"))
        XCTAssertEqual(try WorkflowCodec.encode(document),before)
        let parent=try XCTUnwrap(document.nodes.first {$0.address==root})
        XCTAssertTrue(document.nodes.filter {$0.address.domain == .model && $0.x>parent.x}.allSatisfy {$0.x>=parent.x+1000})
    }
    func testInterfaceEditIsPrivateAndCannotDropConnectedPorts()throws {
        let document=try WorkflowExamples.preparedSeparableConv2dClassifier(),root=WorkflowNodeAddress(domain:.model,localID:"convolution")
        let definition=try WorkflowInlineContainer.definition(in:document,root:root)
        XCTAssertThrowsError(try WorkflowInlineContainer.apply(.setInterface(inputs:[],outputs:definition.outputs),to:document,root:root,registry:.builtIn(),idProvider:{"refuse"}))
        XCTAssertThrowsError(try WorkflowInlineContainer.apply(.setInterface(inputs:definition.inputs,outputs:[]),to:document,root:root,registry:.builtIn(),idProvider:{"refuse"}))
        var outputs=definition.outputs;outputs.append(.init("spatial",source:.init("depthwise","out")))
        let result=try WorkflowInlineContainer.apply(.setInterface(inputs:definition.inputs,outputs:outputs),to:document,root:root,registry:.builtIn(),idProvider:{"ports"})
        XCTAssertEqual(result.edges,document.edges);XCTAssertEqual(result.nodes.count,document.nodes.count)
        XCTAssertEqual(document.definitions,[try SeparableConv2dBlock.definition()])
        let changed=try WorkflowInlineContainer.definition(in:result,root:root)
        XCTAssertEqual(changed.outputs.map(\.id),["out","spatial"]);XCTAssertEqual(changed.body,definition.body)
        XCTAssertTrue(changed.reference.id.hasPrefix("inline-private-"))
    }
    func testInterfaceRejectsMissingAndDuplicateBindings()throws {
        let document=try WorkflowExamples.preparedSeparableConv2dClassifier(),root=WorkflowNodeAddress(domain:.model,localID:"convolution")
        let definition=try WorkflowInlineContainer.definition(in:document,root:root)
        for outputs in [[CompositeOutput("out",source:.init("depthwise","absent"))],definition.outputs+definition.outputs] {
            XCTAssertThrowsError(try WorkflowInlineContainer.apply(.setInterface(inputs:definition.inputs,outputs:outputs),to:document,root:root,registry:.builtIn(),idProvider:{"refuse"}))
        }
    }
}
