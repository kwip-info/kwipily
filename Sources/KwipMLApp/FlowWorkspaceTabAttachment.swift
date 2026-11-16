import KwipMLCore
import AppKit
import SwiftUI

/// Registers the SwiftUI workspace window with the shared file-tab presentation policy.
struct FlowWorkspaceTabAttachment: NSViewRepresentable {
    let title: String
    func makeNSView(context: Context) -> AttachmentView { AttachmentView() }
    func updateNSView(_ view: AttachmentView, context: Context) {
        view.documentTitle = title
        view.registerWindow()
    }
    final class AttachmentView: NSView {
        var documentTitle = "\(ProductBrand.name)"
        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            registerWindow()
        }
        func registerWindow() {
            guard let window else { return }
            FlowCanvasWindowRegistry.shared.registerWorkspaceWindow(window, title: documentTitle)
        }
    }
}
