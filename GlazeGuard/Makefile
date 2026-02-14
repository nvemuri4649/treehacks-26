# Makefile for GlazeGuard

.PHONY: build run clean release test install help

# Default target
all: build

# Build debug version
build:
	@echo "🔨 Building GlazeGuard (debug)..."
	swift build

# Build release version
release:
	@echo "🔨 Building GlazeGuard (release)..."
	swift build -c release
	@echo "✅ Release build complete: .build/release/GlazeGuard"

# Run the app
run: build
	@echo "🚀 Running GlazeGuard..."
	.build/debug/GlazeGuard

# Run tests
test:
	@echo "🧪 Running tests..."
	swift test

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	swift package clean
	rm -rf .build

# Install to /Applications (requires release build)
install: release
	@echo "📦 Installing GlazeGuard to /Applications..."
	@if [ -d "/Applications/GlazeGuard.app" ]; then \
		echo "⚠️  Removing existing installation..."; \
		rm -rf /Applications/GlazeGuard.app; \
	fi
	@echo "Creating app bundle..."
	@mkdir -p GlazeGuard.app/Contents/MacOS
	@mkdir -p GlazeGuard.app/Contents/Resources
	@cp .build/release/GlazeGuard GlazeGuard.app/Contents/MacOS/
	@cp GlazeGuard/Info.plist GlazeGuard.app/Contents/
	@cp -r GlazeGuard/Assets.xcassets GlazeGuard.app/Contents/Resources/
	@cp ../backends.json GlazeGuard.app/Contents/Resources/ 2>/dev/null || true
	@mv GlazeGuard.app /Applications/
	@echo "✅ GlazeGuard installed to /Applications/GlazeGuard.app"

# Open in Xcode
xcode:
	@echo "🔧 Opening in Xcode..."
	open Package.swift

# Check backend server status
check-backend:
	@echo "🔍 Checking backend server..."
	@curl -s http://localhost:8888/health | jq . || echo "❌ Backend not running or not accessible"

# Start backend server (if in project root)
start-backend:
	@echo "🚀 Starting backend server..."
	@cd ../server && python app.py

# Quick development setup
dev-setup:
	@echo "🔧 Setting up development environment..."
	@echo "1. Checking Swift version..."
	@swift --version
	@echo "2. Checking Xcode..."
	@xcodebuild -version
	@echo "3. Verifying backends.json..."
	@test -f ../backends.json && echo "✅ backends.json found" || echo "❌ backends.json missing"
	@echo "4. Checking backend server..."
	@curl -s http://localhost:8888/health > /dev/null && echo "✅ Backend server running" || echo "⚠️  Backend server not running"
	@echo ""
	@echo "Ready to build! Run: make build"

# Show help
help:
	@echo "GlazeGuard Build System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build          - Build debug version"
	@echo "  release        - Build release version"
	@echo "  run            - Build and run debug version"
	@echo "  test           - Run unit tests"
	@echo "  clean          - Remove build artifacts"
	@echo "  install        - Install to /Applications (requires release build)"
	@echo "  xcode          - Open project in Xcode"
	@echo "  check-backend  - Check if backend server is running"
	@echo "  start-backend  - Start the backend server"
	@echo "  dev-setup      - Verify development environment"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make            # Build debug version"
	@echo "  make release    # Build optimized release version"
	@echo "  make run        # Build and run immediately"
	@echo "  make install    # Install to /Applications"
