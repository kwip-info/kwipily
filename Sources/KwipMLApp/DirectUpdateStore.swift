import Foundation
import AppKit
import Combine
import KwipMLNetworking
import KwipMLCore
#if KWIPML_DIRECT_UPDATES
import Sparkle
#endif

@MainActor final class DirectUpdateStore:NSObject,ObservableObject {
    static let shared=DirectUpdateStore()
    @Published private(set) var available=false
    @Published private(set) var canCheck=false
    @Published private(set) var automaticallyChecks=true
    @Published private(set) var diagnosticOutcome=UpdateDiagnosticOutcome.notConfigured
    @Published private(set) var message="Updates are available in the direct-distribution edition."
    private var didStart=false
    // Keep the visible status and exported fixed category in the same transition.
    // Equal outcomes preserve a more specific, fixed refusal through SDK completion.
    func transition(to outcome:UpdateDiagnosticOutcome, message specific:String?=nil) {
        guard outcome != diagnosticOutcome || specific != nil else{return}
        diagnosticOutcome=outcome
        if let specific {message=specific;return}
        switch outcome {
        case .notConfigured: message="Updates are available in the direct-distribution edition."
        case .ready: message="Updates include v1 releases. You choose when to download and install."
        case .checking: message="Checking for a compatible v1 update…"
        case .activeWork: message="Finish or cancel active work before continuing with updates."
        case .signInRequired: message="The update service refused this request. Try again later."
        case .refused: message="The update was refused. Check for updates again or contact contact@kwip.info for help."
        case .offerAvailable: message="A verified v1 update is available. You choose when to download and install."
        case .downloading: message="Downloading the update…"
        case .finished: message="Update cycle finished. Check the app version after reopening to confirm installation."
        case .noUpdate: message="No newer compatible update is available."
        case .cancelled: message="Update cancelled. You can check again when ready."
        case .configurationFailed: message="Updates could not be configured safely. Contact contact@kwip.info for help."
        case .networkFailed: message="The update check or download could not reach the service. Try again when connected."
        case .verificationFailed: message="The update could not be verified. Check again or contact contact@kwip.info for help."
        case .installationFailed: message="The update could not be installed. Keep your project files and contact contact@kwip.info for help."
        case .otherFailure: message="The update did not complete. Try again or contact contact@kwip.info for help."
        }
    }
#if KWIPML_DIRECT_UPDATES
    private var controller:SPUStandardUpdaterController?
    private var cycleAdmission=DirectUpdateCycleAdmission()
    private var observations=Set<AnyCancellable>()
#endif
    func start() {
        guard !didStart else{return};didStart=true
#if KWIPML_DIRECT_UPDATES
        guard DirectUpdatePolicy.configured(Bundle.main.infoDictionary ?? [:]) else {
            transition(to:.configurationFailed,message:"Updates are not configured for this build.");return
        }
        let value=SPUStandardUpdaterController(startingUpdater:false,updaterDelegate:self,userDriverDelegate:nil)
        controller=value
        value.updater.sendsSystemProfile=false
        do {try value.updater.start()} catch {transition(to:.classify(error as NSError),message:"Updates could not start. Contact contact@kwip.info for help.");return}
        available=true
        transition(to:.ready)
        value.updater.publisher(for:\.canCheckForUpdates).receive(on:RunLoop.main).sink{[weak self] in self?.canCheck=$0}.store(in:&observations)
        value.updater.publisher(for:\.automaticallyChecksForUpdates).receive(on:RunLoop.main).sink{[weak self] in self?.automaticallyChecks=$0}.store(in:&observations)
#endif
    }
    func setAutomaticChecks(_ enabled:Bool) {
#if KWIPML_DIRECT_UPDATES
        guard available,let updater=controller?.updater else{return}
        updater.automaticallyChecksForUpdates=enabled
        automaticallyChecks=enabled
#endif
    }
    func check() {
#if KWIPML_DIRECT_UPDATES
        guard available,canCheck,let updater=controller?.updater else{return}
        updater.checkForUpdates()
#endif
    }
}

#if KWIPML_DIRECT_UPDATES
extension DirectUpdateStore:SPUUpdaterDelegate {
    private func refuse(_ text:String, outcome:UpdateDiagnosticOutcome = .refused)throws->Never {
        transition(to:outcome,message:text)
        throw NSError(domain:"info.kwip.KwipML.updates",code:1,userInfo:[NSLocalizedDescriptionKey:text])
    }
    func feedURLString(for updater:SPUUpdater)->String? { DirectUpdatePolicy.feedURL }
    func updater(_ updater:SPUUpdater,mayPerform updateCheck:SPUUpdateCheck)throws {
        updater.httpHeaders=nil
        cycleAdmission.reset()
        guard updater.feedURL?.absoluteString==DirectUpdatePolicy.feedURL else {
            try refuse("The update destination could not be verified.",outcome:.configurationFailed)
        }
        guard !ProjectWindowRegistry.shared.hasActiveOperations else {try refuse("Finish or cancel active work before checking for updates.",outcome:.activeWork)}
        guard DirectUpdatePolicy.canCheck(production:DesktopDeployment.isCommercialEdition(bundle:Bundle.main.bundleIdentifier),configured:available,
          activeWork:ProjectWindowRegistry.shared.hasActiveOperations) else {
            try refuse("Updates are not available while this build or active work prevents a safe check.")
        }
        transition(to:.checking)
    }
    func updater(_ updater:SPUUpdater,didFinishLoading appcast:SUAppcast) {
        cycleAdmission.observe(appcast.items.map(\.propertiesDictionary))
        if !cycleAdmission.metadataValid {transition(to:.verificationFailed,message:"The update metadata could not be verified. Check again or contact contact@kwip.info for help.")}
    }
    func updater(_ updater:SPUUpdater,shouldProceedWithUpdate item:SUAppcastItem,updateCheck:SPUUpdateCheck)throws {
        guard cycleAdmission.metadataValid,let offer=DirectUpdateSparkleAdapter.offer(item) else {
            try refuse("This update could not be verified as a compatible v1 release. Contact contact@kwip.info for help.")
        }
        guard !ProjectWindowRegistry.shared.hasActiveOperations else {try refuse("Finish or cancel active work before installing an update.",outcome:.activeWork)}
        guard cycleAdmission.admit(offer,installedBuild:Bundle.main.object(forInfoDictionaryKey:"CFBundleVersion") as? String ?? "",activeWork:false) else {
            try refuse("This update could not be verified as a compatible v1 release. Contact contact@kwip.info for help.")
        }
        transition(to:.offerAvailable)
    }
    func updater(_ updater:SPUUpdater,willDownloadUpdate item:SUAppcastItem,with request:NSMutableURLRequest) {
        // Both selected delta and later full fallback reach this callback. Strip captured
        // authorization before validating the concrete item, target and destination.
        request.setValue(nil,forHTTPHeaderField:"Authorization")
        guard !ProjectWindowRegistry.shared.hasActiveOperations else {
            request.url=URL(string:"about:blank")
            transition(to:.activeWork,message:"Finish or cancel active work before downloading an update.");return
        }
        guard cycleAdmission.metadataValid,let target=cycleAdmission.target,let offer=DirectUpdateSparkleAdapter.offer(item) else {
            request.url=URL(string:"about:blank")
            transition(to:.verificationFailed,message:"The update download metadata could not be verified. Check for updates again.");return
        }
        let ready=DirectUpdatePolicy.prepareDownloadRequest(request,offer:offer,
            installedBuild:Bundle.main.object(forInfoDictionaryKey:"CFBundleVersion") as? String ?? "",target:target,activeWork:false)
        if ready {transition(to:.downloading)}
        else {transition(to:.configurationFailed,message:"The update download was refused. Check for updates again.")}
    }
    func updater(_ updater:SPUUpdater,shouldDownloadReleaseNotesForUpdate item:SUAppcastItem)->Bool {false}
    func userDidCancelDownload(_ updater:SPUUpdater) {cycleAdmission.reset();transition(to:.cancelled)}
    func updater(_ updater:SPUUpdater,didAbortWithError error:Error) {
        cycleAdmission.reset()
        // Preserve our fixed refusal reason. Never retain or export raw SDK errors.
        transition(to:diagnosticOutcome.completingCycle(error:error as NSError))
    }
    func updater(_ updater:SPUUpdater,didFinishUpdateCycleFor updateCheck:SPUUpdateCheck,error:Error?) {
        cycleAdmission.reset()
        transition(to:diagnosticOutcome.completingCycle(error:error.map{$0 as NSError}))
        // A completed cycle is not proof of installation.
    }
    func updaterShouldRelaunchApplication(_ updater:SPUUpdater)->Bool {
        guard !ProjectWindowRegistry.shared.hasActiveOperations else {
            transition(to:.activeWork,message:"Finish or cancel active work, then check for updates again to install.");return false
        }
        return true
    }
}
#endif

#if KWIPML_DIRECT_UPDATES
/// Sparkle 2.10 exposes the selected delta's original enclosure through this public dictionary.
enum DirectUpdateSparkleAdapter {
    static func offer(_ item:SUAppcastItem)->DirectUpdateOffer? {
        guard let enclosure=item.propertiesDictionary["enclosure"] as? [String:Any] else{return nil}
        let rawBase=enclosure["sparkle:deltaFrom"]
        let base=rawBase as? String
        guard item.isDeltaUpdate == (rawBase != nil),rawBase == nil || base != nil else{return nil}
        let rawSignature=enclosure["sparkle:edSignature"]
        guard rawSignature == nil || rawSignature is String else{return nil}
        return DirectUpdateOffer(version:item.displayVersionString,build:item.versionString,url:item.fileURL,
            size:item.contentLength,signedFeed:item.signingValidationStatus == .succeeded,
            isDelta:item.isDeltaUpdate,deltaFrom:base,artifactSignature:rawSignature as? String)
    }
}
#endif
