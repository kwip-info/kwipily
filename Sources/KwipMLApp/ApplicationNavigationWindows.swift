import KwipMLCore
import AppKit
import SwiftUI

/// App surfaces have one independent owner each; they are never documents or canvas tabs.
@MainActor final class ApplicationNavigationWindows {
    static let shared = ApplicationNavigationWindows()
    private var session: ProjectWindowSession?
    private var settingsController: NSWindowController?
    private var learnController: NSWindowController?
    private var templatesController: ExpandedConfigurationWindowController?
    private var templateSetupControllers:[UUID:ExpandedConfigurationWindowController]=[:]
    let templateNavigation = WorkflowTemplateNavigation()
    let learnNavigation = LearningLibraryNavigation()
    let settingsNavigation = AppSettingsNavigation()
    private let presentsWindows: Bool
    var settingsWindow: NSWindow? { settingsController?.window }
    var learnWindow: NSWindow? { learnController?.window }
    var templatesWindow:NSWindow? {templatesController?.window}
    func templateSetupWindow(sessionID:UUID)->NSWindow? {templateSetupControllers[sessionID]?.window}
    init(presentsWindows: Bool = true) { self.presentsWindows = presentsWindows }
    func configure(session: ProjectWindowSession) { self.session = session }
    func showSettings(page: SettingsPage? = nil) {
        guard let session else { return }
        if let page { settingsNavigation.open(page) }
        if settingsController == nil {
            settingsController = controller(title: "\(ProductBrand.name) · Settings", id: "info.kwip.KwipML.settings", size: .init(width: 960, height: 760),
                content: AppSettingsView(store: session.workspace, navigation: settingsNavigation).appAppearance(session.workspace.appearanceState))
        }
        present(settingsController)
    }
    func showTemplates(id: String? = nil) {
        guard let session else { return }
        if let id { templateNavigation.selectedID = id }
        if templatesController?.didClose != false {
            let navigation=templateNavigation
            templatesController = ExpandedConfigurationWindowController(title: "\(ProductBrand.name) · Templates", identifier: "info.kwip.KwipML.templates") { [weak self] editor in
                AnyView(GeometryReader {viewport in WorkflowTemplateGalleryView(navigation: navigation, create: { [weak self] id in
                    _ = try ProjectWindowRegistry.shared.newFlow(templateID: id)
                    self?.templatesController?.close()
                },close:{[weak editor] in editor?.close()}).frame(width:viewport.size.width,height:viewport.size.height).clipped()
                }.frame(minWidth:800,minHeight:610).windowActivatingControls().appAppearance(session.workspace.appearanceState))
            }
        }
        present(templatesController)
    }
    func showTemplateSetup(workspace:WorkspaceStore) {
        guard let host=workspace.workflowSession,let origin=host.owner.document.templateOrigin,
              let entry=try? WorkflowTemplates.catalog().first(where:{$0.id==origin.id}) else{return}
        templateSetupControllers=templateSetupControllers.filter{!$0.value.didClose}
        let identity=host.owner.sessionID
        if let existing=templateSetupControllers[identity] {if presentsWindows {existing.present()};return}
        let editor=ExpandedConfigurationWindowController(title:"Template · \(entry.title)",identifier:"info.kwip.KwipML.template-setup.\(identity.uuidString)") {controller in
            AnyView(GeometryReader {viewport in WorkflowTemplateSetupView(owner:host.owner,entry:entry,sources:host.sourceProvider,
                revealConfiguration:{workspace.showInspector=true},close:{[weak controller] in controller?.close()},workspace:workspace,
                expanded:true,registerDiscard:{[weak controller] discard,dirty in controller?.discardDraft=discard;controller?.hasUnappliedDraft=dirty})
                .frame(width:viewport.size.width,height:viewport.size.height,alignment:.top).clipped()
            }.frame(minWidth:800,minHeight:610).windowActivatingControls().appAppearance(workspace.appearanceState))
        }
        templateSetupControllers[identity]=editor
        if presentsWindows {editor.present()}
    }
    func closeTemplateSetup(sessionID:UUID) {
        templateSetupControllers[sessionID]?.close()
        if templateSetupControllers[sessionID]?.didClose == true {templateSetupControllers[sessionID]=nil}
    }
    func openLessonOnCanvas(entryID:String,operation:String?=nil) {
        guard let session else {return}
        do { _ = try ProjectWindowRegistry.shared.openLesson(entryID:entryID,operation:operation,source:session) }
        catch {session.workspace.statusMessage="Lesson could not open: \(error.localizedDescription)"}
    }
    func returnToLearningLibrary() {
        learnNavigation.showSection("Library")
        showLearn()
    }
    func showLearn(entryID: String) {
        guard learnNavigation.open(entryID: entryID) else { return }
        showLearn()
    }
    func showLearn(search: String? = nil) {
        guard let session else { return }
        if let search { learnNavigation.open(search: search) }
        session.workspace.refreshProjects()
        if learnController == nil {
            learnController = controller(title: "\(ProductBrand.name) · Learn", id: "info.kwip.KwipML.learn", size: .init(width: 1100, height: 760),
                content: LearningLibraryView(store: session.workspace, affineLearning: session.affine,
                    normalizationLearning: session.normalization, tensorLearning: session.tensor, navigation: learnNavigation)
                    .appAppearance(session.workspace.appearanceState))
        }
        present(learnController)
    }
    private func controller<Content: View>(title: String, id: String, size: NSSize, content: Content) -> NSWindowController {
        let window = NSWindow(contentRect: .init(origin: .zero, size: size), styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        window.title = title; window.identifier = .init(id)
        window.isReleasedWhenClosed = false; window.tabbingMode = .disallowed
        window.minSize = id.hasSuffix("settings") ? .init(width: 840, height: 580) : id.hasSuffix("templates") ? .init(width: 800, height: 540) : .init(width: 1050, height: 700)
        window.contentView = NSHostingView(rootView: content); window.center()
        return NSWindowController(window: window)
    }
    private func present(_ controller: NSWindowController?) {
        guard presentsWindows else { return }
        controller?.showWindow(nil); controller?.window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}
