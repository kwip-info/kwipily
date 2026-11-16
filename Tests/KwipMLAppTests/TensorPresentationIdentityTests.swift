import XCTest
import SwiftUI
import KwipMLCore
@testable import KwipMLApp

final class TensorPresentationIdentityTests:XCTestCase {
    func testMovingBetweenLayersInOneRunCannotReuseTheWrongTensor() {
        let project=UUID().uuidString,session=UUID(),run=UUID()
        func capture(_ node:String)->WorkflowCapturedDataOutput {
            .init(projectID:project,sessionID:session,address:.init(domain:.model,localID:node),runID:run,requestIdentity:String(repeating:"a",count:64),capturedRevision:1,bindingRevision:2,phase:.completed,matchesCurrentConstruction:true,matchesCurrentBindings:true,ports:[:])
        }
        let first=capture("first"),second=capture("second")
        XCTAssertNotEqual(first.tensorRenderIdentity,second.tensorRenderIdentity)
        var a=TensorOutputPreviewView(preview:[:],renderIdentity:first.tensorRenderIdentity)
        var b=TensorOutputPreviewView(preview:[:],renderIdentity:first.tensorRenderIdentity)
        XCTAssertEqual(a,b)
        b.renderIdentity=second.tensorRenderIdentity;XCTAssertNotEqual(a,b)
        b.renderIdentity=first.tensorRenderIdentity;b.accent = .red;XCTAssertNotEqual(a,b)
        a.renderIdentity=nil;b.renderIdentity=nil;XCTAssertNotEqual(a,b,"Unqualified callers must still refresh their values")
    }
}
