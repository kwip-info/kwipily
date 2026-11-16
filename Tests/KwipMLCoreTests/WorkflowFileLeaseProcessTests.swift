import XCTest
import Foundation
import Darwin
@testable import KwipMLCore

final class WorkflowFileLeaseProcessTests:XCTestCase {
    func testIndependentWorkerHoldsActivityUntilObservedExit()throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python=repo.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath:python.path) else {throw XCTSkip("Development Python required for actual worker lifetime test.")}
        var buffer=[CChar](repeating:0,count:Int(PATH_MAX))
        guard realpath(FileManager.default.temporaryDirectory.path,&buffer) != nil else {throw NSError(domain:NSPOSIXErrorDomain,code:Int(errno))}
        let directory=URL(fileURLWithPath:String(cString:buffer)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
        defer {try? FileManager.default.removeItem(at:directory)}
        let created=try WorkflowFileLease(directory:directory,name:"activity.lock",mode:.exclusive,create:true)
        let identity=created.identity;created.close()
        let process=Process(),output=Pipe(),lifetime=Pipe()
        process.executableURL=python
        process.arguments=["-c","import fcntl,os,signal,sys\nsignal.alarm(5)\nf=os.open(sys.argv[1],os.O_RDWR)\nfcntl.flock(f,fcntl.LOCK_SH|fcntl.LOCK_NB)\nprint('READY',flush=True)\nsys.stdin.buffer.read(1)",directory.appendingPathComponent("activity.lock").path]
        process.standardOutput=output;process.standardError=FileHandle.nullDevice;process.standardInput=lifetime
        try process.run()
        defer {try? lifetime.fileHandleForWriting.close();if process.isRunning {process.terminate()};process.waitUntilExit()}
        let acknowledgement=String(decoding:output.fileHandleForReading.availableData,as:UTF8.self)
        guard acknowledgement=="READY\n" else {return XCTFail("Worker did not establish its activity lock: \(acknowledgement)")}
        XCTAssertTrue(process.isRunning)
        XCTAssertThrowsError(try WorkflowFileLease(directory:directory,name:"activity.lock",mode:.exclusive,expectedIdentity:identity)) {XCTAssertEqual($0 as? WorkflowFileLeaseError,.busy)}
        // Closing a parent channel is not itself proof of exit. Wait for this exact child.
        try lifetime.fileHandleForWriting.close()
        process.waitUntilExit()
        XCTAssertEqual(process.terminationReason,.exit);XCTAssertEqual(process.terminationStatus,0)
        let recovery=try WorkflowFileLease(directory:directory,name:"activity.lock",mode:.exclusive,expectedIdentity:identity)
        XCTAssertNoThrow(try recovery.checkIdentity())
    }
}
