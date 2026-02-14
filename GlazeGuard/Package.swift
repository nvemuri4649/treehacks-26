// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "GlazeGuard",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(
            name: "GlazeGuard",
            targets: ["GlazeGuard"]
        )
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "GlazeGuard",
            path: "GlazeGuard",
            resources: [
                .process("Assets.xcassets")
            ],
            swiftSettings: [
                .enableUpcomingFeature("BareSlashRegexLiterals")
            ]
        )
    ]
)
