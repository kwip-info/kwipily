import SwiftUI
import UniformTypeIdentifiers

/// Tracks palette drags across container frames without adding a gesture surface over children.
struct CanvasInlineContainerDropDelegate:DropDelegate {
    let update:@MainActor (CGPoint)->Void
    let exit:@MainActor ()->Void
    let perform:@MainActor (String,CGPoint)->Bool
    private var types:[String] {[UTType.utf8PlainText.identifier,UTType.text.identifier]}
    func validateDrop(info:DropInfo)->Bool {info.hasItemsConforming(to:types)}
    func dropEntered(info:DropInfo) {MainActor.assumeIsolated {update(info.location)}}
    func dropUpdated(info:DropInfo)->DropProposal? {
        MainActor.assumeIsolated {update(info.location)}
        return DropProposal(operation:.copy)
    }
    func dropExited(info:DropInfo) {MainActor.assumeIsolated {exit()}}
    func performDrop(info:DropInfo)->Bool {
        MainActor.assumeIsolated {exit()}
        guard let provider=info.itemProviders(for:types).first,provider.canLoadObject(ofClass:NSString.self) else {return false}
        let point=info.location
        provider.loadObject(ofClass:NSString.self) {value,error in
            guard error == nil,let text=value as? String else {return}
            Task {@MainActor in _=perform(text,point)}
        }
        return true
    }
}
