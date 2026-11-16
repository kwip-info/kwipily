import Foundation
import Combine
import CryptoKit
import Darwin
import KwipMLCore

/// Observational evidence only. Failure here never changes the training checkpoint's status.
@MainActor final class GradientCaptureStore:ObservableObject {
    enum Status:Equatable { case off,waiting,capturing,partial,verified,invalid }
    @Published private(set) var status:Status = .off
    @Published private(set) var message="Gradient capture is off."
    @Published private(set) var supportedDevices:[String]=[]
    @Published private(set) var requestedConfiguration:GradientCaptureConfiguration?
    @Published private(set) var runID:String?
    @Published private(set) var device:String?
    @Published private(set) var inventory:GradientCaptureInventory?
    @Published private(set) var summaries:[GradientEpochSummary]=[]
    @Published private(set) var artifact:GradientCaptureArtifact?
    var canCapture:Bool {!supportedDevices.isEmpty}
    private var initialEpoch=0
    private var finalEpoch=0
    private var trainingRows=0
    private var batchSize=0

    func installCapability(_ data:Data?) {
        supportedDevices=[]
        guard let data,let object=try? Self.object(data,limit:8192),Set(object.keys)==["schemaVersion","id","supportedDevices","phase","limits"],
              object["schemaVersion"] == .integer(1),object["id"] == .string("kwip.gradient-summary.v1"),object["phase"] == .string("afterBackwardBeforeStep"),
              case .array(let devices)=object["supportedDevices"],!devices.isEmpty,devices.count<=2,
              case .object(let limits)=object["limits"],limits == ["epochs":10000,"batchesPerEpoch":8,"captures":1024,"artifactBytes":8000000,"chunkElements":65536] else {return}
        let names=devices.compactMap(\.stringValue)
        guard names.count==devices.count,Set(names).count==names.count,Set(names).isSubset(of:["cpu","mps"]) else {return}
        supportedDevices=names
    }
    func reset() {
        requestedConfiguration=nil;runID=nil;device=nil;inventory=nil;summaries=[];artifact=nil
        initialEpoch=0;finalEpoch=0;trainingRows=0;batchSize=0;status = .off;message="Gradient capture is off."
    }
    func begin(config:GradientCaptureConfiguration?) {
        reset();requestedConfiguration=config
        guard let config else {return}
        do {
            guard canCapture else {throw Failure("This runtime has no verified gradient capture capability.")}
            _ = try GradientCaptureCodec.decodeConfiguration(JSONEncoder().encode(config))
            status = .waiting;message="Waiting for gradient capture inventory."
        } catch {reject(error)}
    }
    func consumeStarted(_ data:Data) {
        guard (status == .waiting || status == .off),runID == nil else {if status != .invalid {reject(Failure("Duplicate or unexpected gradient start."))};return}
        do {
            let event=try Self.object(data)
            guard event["event"] == .string("started"),let id=event["runID"]?.stringValue,UUID(uuidString:id) != nil,
                  let actualDevice=event["device"]?.stringValue else {throw Failure("Invalid gradient run identity.")}
            runID=id;device=actualDevice
            guard let requestedConfiguration else {
                guard event["gradientCapture"] == nil else {throw Failure("Unrequested gradient capture was reported.")}
                return
            }
            guard supportedDevices.contains(actualDevice),let raw=event["gradientCapture"] else {throw Failure("Requested gradient capture inventory is missing or unsupported.")}
            let started=try GradientCaptureCodec.decodeStarted(JSONEncoder().encode(raw))
            guard started.config==requestedConfiguration,let end=event["totalEpochs"]?.integerValue,end>started.initialEpoch,end<=10000,
                  event["initialEpoch"]?.integerValue==started.initialEpoch,
                  let rows=event["trainingSamples"]?.integerValue,rows>0,rows<=100_000,
                  let batch=event["batchSize"]?.integerValue,batch>0,batch<=rows else {throw Failure("Gradient capture start differs from the requested configuration or training bounds.")}
            inventory=started.inventory;initialEpoch=started.initialEpoch;finalEpoch=end;trainingRows=rows;batchSize=batch
            status = .capturing;message="Live gradient summaries · incomplete until the run is verified."
        } catch {reject(error)}
    }
    func consumeSummary(_ data:Data) {
        guard status != .invalid else {return}
        do {
            guard status == .capturing,let runID,let requestedConfiguration,let inventory else {throw Failure("Gradient summary arrived outside an enabled run.")}
            let value=try GradientCaptureCodec.decodeEvent(data,runID:runID,config:requestedConfiguration,inventory:inventory,previous:summaries.last,initialEpoch:initialEpoch)
            guard value.epoch<=finalEpoch,value.totalTrainingRows==trainingRows,value.totalBatches==(trainingRows+batchSize-1)/batchSize,
                  summaries.count<10000 else {throw Failure("Gradient summary differs from the started training coverage.")}
            for batch in value.batches {
                guard batch.trainingRows==min(batchSize,trainingRows-batch.batchIndex*batchSize) else {throw Failure("Gradient batch row coverage differs from the started minibatches.")}
            }
            summaries.append(value)
        } catch {reject(error)}
    }
    func interrupt(status reason:String) {
        guard (status == .waiting || status == .capturing),requestedConfiguration != nil else {return}
        artifact=nil;status = .partial;message="\(String(reason.prefix(200))) · partial gradient observations; no completed report."
    }
    func complete(directory:URL,manifestData:Data,completedMetadata:Data?=nil) {
        guard status != .invalid else {return}
        do {
            let loaded=try readCompleted(directory:directory,manifestData:manifestData)
            guard let loaded else {
                guard requestedConfiguration == nil else {throw Failure("Completed run omitted requested gradient capture.")}
                status = .off;message="This run did not capture gradients.";return
            }
            guard status == .capturing,let completedMetadata,
                  try GradientCaptureCodec.decodeMetadata(completedMetadata)==loaded.metadata,
                  loaded.artifact.runID==runID,loaded.artifact.device==device,
                  loaded.artifact.config==requestedConfiguration,loaded.artifact.inventory==inventory,
                  loaded.artifact.initialEpoch==initialEpoch,loaded.artifact.finalEpoch==finalEpoch,
                  loaded.artifact.summaries==summaries else {throw Failure("Completed gradient evidence differs from the live run or completion metadata.")}
            install(loaded.artifact)
        } catch {reject(error)}
    }
    func restore(directory:URL,manifestData:Data) {
        reset()
        do {
            guard let loaded=try readCompleted(directory:directory,manifestData:manifestData) else {message="This saved run did not capture gradients.";return}
            install(loaded.artifact)
        } catch {reject(error)}
    }
    private func install(_ value:GradientCaptureArtifact) {
        artifact=value;runID=value.runID;device=value.device;inventory=value.inventory;requestedConfiguration=value.config;summaries=value.summaries
        status = .verified;message="Verified saved gradient summaries · after backward, before optimizer step."
    }
    private func reject(_ error:Error) {
        summaries=[];artifact=nil;inventory=nil;status = .invalid
        message="Gradient evidence unavailable: " + String(error.localizedDescription.prefix(500))
    }
    private func readCompleted(directory:URL,manifestData:Data)throws->(artifact:GradientCaptureArtifact,metadata:GradientCaptureMetadata)? {
        let manifest=try Self.object(manifestData,limit:20_000_000)
        guard manifest["status"] == .string("completed"),let id=manifest["runID"]?.stringValue,UUID(uuidString:id) != nil,
              case .object(let runtime)=manifest["runtime"],let device=runtime["device"]?.stringValue,["cpu","mps"].contains(device) else {throw Failure("Gradient evidence requires a completed run manifest with an exact device and run identity.")}
        guard let raw=manifest["gradientCapture"] else {return nil}
        let metadata=try GradientCaptureCodec.decodeMetadata(JSONEncoder().encode(raw))
        let bytes=try GradientCaptureFileReader.read(directory:directory)
        guard bytes.count==metadata.bytes,SHA256.hash(data:bytes).map({String(format:"%02x",$0)}).joined()==metadata.sha256 else {throw Failure("Saved gradient artifact hash or byte count differs from its manifest.")}
        return (try GradientCaptureCodec.decodeArtifact(bytes,expectedRunID:id,expectedDevice:device,metadata:metadata),metadata)
    }
    private static func object(_ data:Data,limit:Int=1_000_000)throws->[String:JSONValue] {
        guard data.count<=limit,case .object(let object)=try JSONDecoder().decode(JSONValue.self,from:data) else {throw Failure("Gradient data must be a bounded JSON object.")}
        return object
    }
    private struct Failure:LocalizedError {let message:String;init(_ message:String){self.message=message};var errorDescription:String?{message}}
}

/// Fixed sibling path, descriptor-relative no-follow read; hash verification uses these same bytes.
private enum GradientCaptureFileReader {
    static func read(directory:URL)throws->Data {
        let failure=NSError(domain:"KwipML.GradientCapture",code:1,userInfo:[NSLocalizedDescriptionKey:"Gradient summary must be a bounded regular file in the run directory."])
        let parent=Darwin.open(directory.path,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)
        guard parent>=0 else {throw failure};defer{Darwin.close(parent)}
        let fd=openat(parent,"gradient-summary.json",O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC)
        guard fd>=0 else {throw failure};defer{Darwin.close(fd)}
        var before=stat()
        guard fstat(fd,&before)==0,before.st_mode & mode_t(S_IFMT)==mode_t(S_IFREG),before.st_size>=0,before.st_size<=8_000_000 else {throw failure}
        var result=Data(),buffer=[UInt8](repeating:0,count:65536)
        while result.count<=8_000_000 {
            let count=buffer.withUnsafeMutableBytes {Darwin.read(fd,$0.baseAddress!,min($0.count,8_000_001-result.count))}
            if count<0 {if errno==EINTR {continue};throw failure};if count==0 {break}
            result.append(contentsOf:buffer.prefix(count))
        }
        var after=stat()
        guard result.count<=8_000_000,fstat(fd,&after)==0,before.st_size==after.st_size,before.st_mtimespec.tv_sec==after.st_mtimespec.tv_sec,before.st_mtimespec.tv_nsec==after.st_mtimespec.tv_nsec else {throw failure}
        return result
    }
}
