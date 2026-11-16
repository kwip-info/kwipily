import XCTest
import SwiftUI
import AppKit
@testable import KwipMLApp

/// Verification of existing root integration, not a synthetic policy-only test.
/// The hidden test window never becomes key or changes system accessibility settings.
@MainActor final class AppearanceReducedMotionHostingTests: XCTestCase {
    func testAppearanceRootSuppressesAnimatedTransactionsAndPreservesHostedViewIdentity() async throws {
        // SwiftUI exposes this accessibility environment value read-only. Each
        // invocation checks the real OS state; run separately with it off/on.
        let reduced=NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
        let model=MotionProbeModel(),appearance=AppAppearanceState()
        let recorder=MotionTransactionRecorder()
        let initial=expectation(description:"Hosted initial transaction reduced=\(reduced)")
        recorder.wait(for:0,expectation:initial)
        let content=MotionProbeContainer(model:model,appearance:appearance,recorder:recorder)
        let hosting=NSHostingView(rootView:content)
        let window=NSWindow(contentRect:NSRect(x:0,y:0,width:240,height:100),styleMask:.borderless,backing:.buffered,defer:false)
        window.isReleasedWhenClosed=false
        window.contentView=hosting
        hosting.frame=NSRect(x:0,y:0,width:240,height:100)
        defer {window.contentView=nil;window.close()}
        hosting.layoutSubtreeIfNeeded()
        await fulfillment(of:[initial],timeout:5)
        let original=try XCTUnwrap(recorder.samples.first(where:{$0.value == 0}))

        let updated=expectation(description:"Hosted explicit animation reduced=\(reduced)")
        recorder.wait(for:1,expectation:updated)
        withAnimation(.linear(duration:0.25)) { model.value=1 }
        hosting.layoutSubtreeIfNeeded()
        await fulfillment(of:[updated],timeout:5)
        let first=try XCTUnwrap(recorder.samples.first(where:{$0.value == 1}))
        XCTAssertEqual(first.viewIdentity,original.viewIdentity)
        XCTAssertEqual(first.animationIsNil,reduced,"The non-reduced baseline must receive the animation to prove a real transaction reached the probe")
        XCTAssertEqual(first.disablesAnimations,reduced)
        XCTAssertEqual(first.reduceMotion,reduced)

        let nested=expectation(description:"Hosted nested animation reduced=\(reduced)")
        recorder.wait(for:2,expectation:nested)
        withAnimation(.easeInOut(duration:0.5)) {
            withAnimation(.linear(duration:0.25)) { model.value=2 }
        }
        hosting.layoutSubtreeIfNeeded()
        await fulfillment(of:[nested],timeout:5)
        let second=try XCTUnwrap(recorder.samples.first(where:{$0.value == 2}))
        XCTAssertEqual(second.viewIdentity,original.viewIdentity)
        XCTAssertEqual(second.animationIsNil,reduced)
        XCTAssertEqual(second.disablesAnimations,reduced)
        XCTAssertEqual(model.value,2,"Appearance policy must not discard the user's state update")
    }
}

@MainActor private final class MotionProbeModel:ObservableObject {
    @Published var value=0
}
@MainActor private final class MotionTransactionRecorder {
    struct Sample {
        let value:Int
        let viewIdentity:ObjectIdentifier
        let animationIsNil:Bool
        let disablesAnimations:Bool
        let reduceMotion:Bool
    }
    private(set) var samples:[Sample]=[]
    private var waiting:[Int:XCTestExpectation]=[:]
    func wait(for value:Int,expectation:XCTestExpectation) {waiting[value]=expectation}
    func record(_ sample:Sample) {
        samples.append(sample)
        waiting.removeValue(forKey:sample.value)?.fulfill()
    }
}
private struct MotionProbeContainer:View {
    @ObservedObject var model:MotionProbeModel
    let appearance:AppAppearanceState
    let recorder:MotionTransactionRecorder
    var body:some View {
        MotionTransactionProbe(value:model.value,recorder:recorder)
            .frame(width:220,height:80)
            .appAppearance(appearance)
    }
}
private struct MotionTransactionProbe:NSViewRepresentable {
    let value:Int
    let recorder:MotionTransactionRecorder
    func makeNSView(context:Context)->NSView {NSView(frame:.zero)}
    func updateNSView(_ view:NSView,context:Context) {
        recorder.record(.init(value:value,viewIdentity:ObjectIdentifier(view),
            animationIsNil:context.transaction.animation == nil,
            disablesAnimations:context.transaction.disablesAnimations,
            reduceMotion:context.environment.accessibilityReduceMotion))
    }
}
